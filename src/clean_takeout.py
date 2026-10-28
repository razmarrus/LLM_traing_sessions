"""Extract my own sent messages from a Google Takeout Gmail mbox into notebook JSONL."""

import argparse
import csv
import json
import logging
import mailbox
import re
from collections import Counter
from collections.abc import Iterable, Iterator
from email import policy
from email.message import EmailMessage
from email.parser import BytesHeaderParser, BytesParser
from html.parser import HTMLParser
from pathlib import Path

logger = logging.getLogger(__name__)

PARSER = BytesParser(policy=policy.default)
HEADER_PARSER = BytesHeaderParser(policy=policy.default)
HEADER_END_RE = re.compile(rb"\r?\n\r?\n")

KEPT = "kept"
DRAFT_LABELS = {"draft", "drafts"}

# HTML: stop at the reply chain, skip quotes and signatures, break lines at blocks
STOP_CLASSES = {"gmail_quote", "gmail_quote_container", "gmail_extra"}
STOP_IDS = {
    "divrplyfwdmsg", "appendonsend", "mail-editor-reference-message-container",
}
SKIP_CLASSES = {"gmail_signature", "gmail_signature_prefix", "moz-signature"}
SKIP_TAGS = {"blockquote", "head", "style", "script", "title"}
BLOCK_TAGS = {
    "p", "div", "li", "tr", "table", "ul", "ol", "h1", "h2", "h3", "h4", "h5", "h6",
}
VOID_TAGS = {
    "br", "img", "hr", "meta", "link", "input", "wbr", "col", "area", "base", "source",
}
# Source whitespace collapses in HTML; no-break spaces are visible and kept
HTML_SPACE_RE = re.compile(r"[ \t\r\n\f]+")

# Reply headers can wrap onto one extra line (long names and addresses)
WRAP = r"[^\n]{0,300}?(?:\n[^\n]{0,300}?)?"

# Everything from the first match onward is quoted, forwarded or a signature
CUTOFF_PATTERNS = [
    rf"^On {WRAP}wrote:[ \t]*$",
    rf"^Op {WRAP}(?:schreef|geschreven){WRAP}:[ \t]*$",
    rf"^Am {WRAP}schrieb{WRAP}:[ \t]*$",
    rf"^Le {WRAP}a écrit[ \t]*:[ \t]*$",
    rf"^El {WRAP}escribió:[ \t]*$",
    rf"^Il giorno {WRAP}ha scritto:[ \t]*$",
    # Forwards and Outlook-style quoted headers
    r"^-{5,}\s*(Forwarded message|Doorgestuurd bericht|Original Message|"
    r"Oorspronkelijk bericht|Ursprüngliche Nachricht)\s*-{5,}",
    r"^Begin forwarded message:",
    r"^_{20,}\s*$",
    r"^(From|Van|Von|De): .+\n(Sent|Date|Verzonden|Datum|Gesendet|Envoyé): ",
    # Signature delimiter (RFC 3676) and mobile client footers
    r"^-- ?$",
    r"^(Sent from|Verzonden vanaf|Verstuurd vanaf|Gesendet von) "
    r"(my|mijn|meinem) .{0,40}$",
    r"^Get Outlook for .{0,30}$",
]
CUTOFF_RE = re.compile(
    "|".join(f"(?:{p})" for p in CUTOFF_PATTERNS), re.M | re.I
)
QUOTED_LINE_RE = re.compile(r"^>.*$\n?", re.M)
MBOXO_FROM_RE = re.compile(r"^>(>*From )", re.M)
BLANK_LINES_RE = re.compile(r"\n{3,}")
LINE_EDGE_SPACE_RE = re.compile(r"[ \t]*\n[ \t]*")
CHAR_MAP = str.maketrans({
    "\r": None,
    "\N{NO-BREAK SPACE}": " ",
    "\N{ZERO WIDTH SPACE}": None,
    "\N{ZERO WIDTH NO-BREAK SPACE}": None,
})


# Reading


def iter_raw_messages(path: Path) -> Iterator[bytes]:
    """Yield raw messages one at a time from an mbox file."""
    box = mailbox.mbox(path, create=False)
    try:
        for key in box.iterkeys():
            yield box.get_bytes(key)
    finally:
        box.close()


def read_labels(raw: bytes) -> set[str]:
    """Return lowercase Takeout labels, parsing only the header block."""
    headers = HEADER_PARSER.parsebytes(HEADER_END_RE.split(raw, maxsplit=1)[0])
    value = str(headers.get("X-Gmail-Labels", ""))
    labels = (label.strip().lower() for label in next(csv.reader([value]), []))
    return {label for label in labels if label}


# Filtering


def label_skip_reason(labels: set[str], label: str) -> str | None:
    """Return why the labels rule a message out, or None."""
    if label.lower() not in labels:
        return "not_sent"
    if labels & DRAFT_LABELS:
        return "draft"
    return None


def content_skip_reason(msg: EmailMessage) -> str | None:
    """Return why a sent message is not my own writing, or None."""
    if str(msg.get("Auto-Submitted", "no")).lower() != "no":
        return "auto_submitted"
    if any(part.get_content_type() == "text/calendar" for part in msg.walk()):
        return "calendar"
    return None


# Body extraction


class VisibleTextParser(HTMLParser):
    """Collect visible text; skip quotes and signatures, stop at the reply chain."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.skip_depth = 0
        self.stopped = False
        self.at_line_start = True

    @property
    def active(self) -> bool:
        """Whether text at the current position is kept."""
        return not (self.stopped or self.skip_depth)

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        """Stop at reply containers, enter skipped subtrees, break lines at blocks."""
        if self.stopped:
            return
        attr = dict(attrs)
        classes = set((attr.get("class") or "").lower().split())
        if classes & STOP_CLASSES or (attr.get("id") or "").lower() in STOP_IDS:
            self.stopped = True
        elif tag in VOID_TAGS:
            self.handle_startendtag(tag, attrs)
        elif self.skip_depth or tag in SKIP_TAGS or classes & SKIP_CLASSES:
            self.skip_depth += 1
        elif tag in BLOCK_TAGS:
            self._end_line()

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        """Turn <br> into a newline; other void tags carry no text."""
        if tag == "br" and self.active:
            self._append("\n")

    def handle_endtag(self, tag: str) -> None:
        """Leave skipped subtrees and break lines at block ends."""
        if self.stopped or tag in VOID_TAGS:
            return
        if self.skip_depth:
            self.skip_depth -= 1
        elif tag in BLOCK_TAGS:
            self._end_line()

    def handle_data(self, data: str) -> None:
        """Keep visible text with source whitespace collapsed."""
        if self.active:
            self._append(HTML_SPACE_RE.sub(" ", data))

    def _append(self, text: str) -> None:
        """Add text and track whether the current line has visible content."""
        self.parts.append(text)
        if text == "\n":
            self.at_line_start = True
        elif text.strip(" "):
            self.at_line_start = False

    def _end_line(self) -> None:
        """Start a new line unless already at one, so nested blocks add no blanks."""
        if not self.at_line_start:
            self._append("\n")


def html_to_text(html: str) -> str:
    """Convert an HTML body to plain text without quoted or signature blocks."""
    parser = VisibleTextParser()
    parser.feed(html)
    parser.close()
    return LINE_EDGE_SPACE_RE.sub("\n", "".join(parser.parts))


def decode_part(part: EmailMessage) -> str:
    """Decode a text part, falling back to replacement chars on bad charsets."""
    try:
        return part.get_content()
    except (LookupError, UnicodeDecodeError):
        payload = part.get_payload(decode=True) or b""
        return payload.decode("utf-8", errors="replace")


def extract_body(msg: EmailMessage) -> str:
    """Return the body as text; HTML first, since Gmail marks quotes there."""
    part = msg.get_body(preferencelist=("html", "plain"))
    if part is None:
        return ""
    content = decode_part(part)
    if part.get_content_type() == "text/html":
        return html_to_text(content)
    return MBOXO_FROM_RE.sub(r"\1", content)


# Text cleaning


def clean_text(text: str) -> str:
    """Cut quoted and signature content, drop quoted lines, normalize whitespace."""
    text = text.translate(CHAR_MAP)
    match = CUTOFF_RE.search(text)
    if match:
        text = text[: match.start()]
    text = QUOTED_LINE_RE.sub("", text)
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    return BLANK_LINES_RE.sub("\n\n", text).strip()


# Pipeline


def process_message(raw: bytes, label: str, min_chars: int) -> tuple[str, str | None]:
    """Return (KEPT, cleaned text) or (skip reason, None) for one raw message."""
    reason = label_skip_reason(read_labels(raw), label)
    if reason:
        return reason, None
    msg = PARSER.parsebytes(raw)
    reason = content_skip_reason(msg)
    if reason:
        return reason, None
    text = clean_text(extract_body(msg))
    if len(text) < min_chars:
        return "empty_after_cleaning", None
    return KEPT, text


def extract_sent(path: Path, label: str, min_chars: int) -> Iterator[str]:
    """Yield cleaned texts of my sent messages and log counts per outcome."""
    counts: Counter[str] = Counter()
    for i, raw in enumerate(iter_raw_messages(path)):
        try:
            outcome, text = process_message(raw, label, min_chars)
        except Exception:  # one malformed message should not stop the export
            logger.exception("Skipping message %d: failed to parse", i)
            outcome, text = "parse_error", None
        counts[outcome] += 1
        if text is not None:
            yield text
    logger.info("Message counts: %s", dict(counts))


def write_jsonl(texts: Iterable[str], output: Path) -> int:
    """Write one {"text": ...} object per line; return the number written."""
    output.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with output.open("w", encoding="utf-8") as f:
        for n, text in enumerate(texts, start=1):
            f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
    return n


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mbox", type=Path, help="Takeout .mbox file")
    parser.add_argument(
        "output", type=Path, help="Output JSONL; keep it outside this repo"
    )
    parser.add_argument(
        "--label", default="Sent", help="Gmail label of my messages (default: Sent)"
    )
    parser.add_argument(
        "--min-chars", type=int, default=1,
        help="Drop messages shorter than this after cleaning",
    )
    return parser.parse_args()


def main() -> None:
    """Run the extraction from the command line."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    args = parse_args()
    n = write_jsonl(extract_sent(args.mbox, args.label, args.min_chars), args.output)
    logger.info("Wrote %d messages to %s", n, args.output)


if __name__ == "__main__":
    main()
