"""Data quality checks for a cleaned messages JSONL file from clean_takeout.py.

Run against a real file:
    STYLE_LORA_MESSAGES=/path/to/my_messages.jsonl poetry run pytest tests/

Failures report line numbers only, never message text, so test logs stay free of
personal data. Inspect flagged lines locally, e.g. `sed -n '<n>p' <file>`.
"""

import json
import os
import re
from collections.abc import Callable
from pathlib import Path

import pytest

from clean_takeout import CHAR_MAP, CUTOFF_RE

MIN_RECOMMENDED_ROWS = 500  # same threshold as the notebook
MAX_CHARS = 10_000  # longer is almost always leftover quoted or forwarded mail
MAX_DUPLICATE_SHARE = 0.2
MAX_REPORTED_LINES = 20
INVISIBLE_CHARS = "".join(map(chr, CHAR_MAP))  # chars the cleaner removes

LEFTOVER_PATTERNS: dict[str, re.Pattern[str]] = {
    "html_tag": re.compile(
        r"</?(p|div|span|br|a|img|table|tr|td|ul|li|b|u|blockquote|html|body)\b[^>]*>",
        re.I,
    ),
    "html_entity": re.compile(r"&(#\d+|#x[0-9a-f]+|[a-z]+);", re.I),
    "quoted_printable": re.compile(r"=(3D|C2=A0|E2=80)|=\n"),
    "mime_header": re.compile(
        r"^(Content-(Type|Transfer-Encoding|Disposition|ID)|MIME-Version|"
        r"X-Gmail-Labels|X-Attachment-Id|Message-ID):",
        re.M | re.I,
    ),
    "mime_boundary": re.compile(r"^--[0-9a-f]{20,}(--)?$", re.M),
    "base64_blob": re.compile(r"^[A-Za-z0-9+/]{60,}={0,2}$", re.M),
    "mbox_from_line": re.compile(r"^From \S+@\S+ \w{3} \w{3} ", re.M),
    "quoted_line": re.compile(r"^>", re.M),
    "reply_or_signature": CUTOFF_RE,
    "invisible_char": re.compile(f"[{re.escape(INVISIBLE_CHARS)}]"),
}


@pytest.fixture(scope="module")
def messages_path() -> Path:
    """Path of the cleaned file from STYLE_LORA_MESSAGES; skip when unset."""
    value = os.environ.get("STYLE_LORA_MESSAGES", "").strip()
    if not value:
        pytest.skip("STYLE_LORA_MESSAGES not set")
    return Path(value)


@pytest.fixture(scope="module")
def raw_lines(messages_path: Path) -> list[str]:
    """All lines of the file, decoded strictly as UTF-8."""
    assert messages_path.is_file(), f"{messages_path} does not exist"
    return messages_path.read_bytes().decode("utf-8").splitlines()


@pytest.fixture(scope="module")
def texts(raw_lines: list[str]) -> dict[int, str]:
    """Map of 1-based line number to message text, for lines that parse."""
    parsed = {}
    for n, line in enumerate(raw_lines, start=1):
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and isinstance(obj.get("text"), str):
            parsed[n] = obj["text"]
    return parsed


def failing_lines(
    texts: dict[int, str], is_bad: Callable[[str], bool]
) -> list[int]:
    """Line numbers whose text fails a check."""
    return [n for n, text in texts.items() if is_bad(text)]


def report(lines: list[int]) -> str:
    """Short failure message with line numbers only."""
    shown = ", ".join(map(str, lines[:MAX_REPORTED_LINES]))
    extra = len(lines) - MAX_REPORTED_LINES
    return f"{len(lines)} lines: {shown}" + (f" (+{extra} more)" if extra > 0 else "")


def test_file_not_empty(raw_lines: list[str]) -> None:
    """File has at least one line."""
    assert raw_lines, "file is empty"


def test_every_line_is_object_with_text(
    raw_lines: list[str], texts: dict[int, str]
) -> None:
    """Each line is a JSON object with a string 'text' field, as the notebook needs."""
    bad = [n for n in range(1, len(raw_lines) + 1) if n not in texts]
    assert not bad, f"invalid JSON or missing 'text': {report(bad)}"


def test_enough_messages(texts: dict[int, str]) -> None:
    """Enough messages to train without heavy overfitting."""
    assert len(texts) >= MIN_RECOMMENDED_ROWS, (
        f"{len(texts)} messages, notebook recommends >= {MIN_RECOMMENDED_ROWS}"
    )


def test_texts_non_empty_and_stripped(texts: dict[int, str]) -> None:
    """No empty messages and no leading or trailing whitespace."""
    bad = failing_lines(texts, lambda t: not t or t != t.strip())
    assert not bad, f"empty or unstripped: {report(bad)}"


def test_no_excess_blank_lines(texts: dict[int, str]) -> None:
    """At most one blank line between paragraphs."""
    bad = failing_lines(texts, lambda t: "\n\n\n" in t)
    assert not bad, f"3+ consecutive newlines: {report(bad)}"


def test_length_within_limit(texts: dict[int, str]) -> None:
    """No message so long that it likely still contains quoted mail."""
    bad = failing_lines(texts, lambda t: len(t) > MAX_CHARS)
    assert not bad, f"longer than {MAX_CHARS} chars: {report(bad)}"


@pytest.mark.parametrize("name", LEFTOVER_PATTERNS)
def test_no_email_leftovers(texts: dict[int, str], name: str) -> None:
    """No HTML, MIME, encoding, quote or signature residue from the export."""
    pattern = LEFTOVER_PATTERNS[name]
    bad = failing_lines(texts, lambda t: pattern.search(t) is not None)
    assert not bad, f"{name}: {report(bad)}"


def test_duplicate_share(texts: dict[int, str]) -> None:
    """Few exact duplicates; many usually means template or auto-generated mail."""
    values = list(texts.values())
    share = 1 - len(set(values)) / len(values) if values else 0.0
    assert share <= MAX_DUPLICATE_SHARE, (
        f"{share:.1%} exact duplicates (max {MAX_DUPLICATE_SHARE:.0%})"
    )
