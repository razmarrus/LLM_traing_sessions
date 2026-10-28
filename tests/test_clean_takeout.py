"""Unit tests for the critical paths of clean_takeout.py."""

import json
import mailbox
from email.message import EmailMessage
from pathlib import Path

import pytest

from clean_takeout import (
    KEPT,
    clean_text,
    extract_sent,
    html_to_text,
    process_message,
    write_jsonl,
)


def make_message(
    labels: str = "Sent",
    plain: str | None = "Hello",
    html: str | None = None,
    calendar: bool = False,
    pdf: bool = False,
) -> EmailMessage:
    """Build a Takeout-like message with optional HTML, calendar and PDF parts."""
    msg = EmailMessage()
    msg["From"] = "me@example.com"
    msg["X-Gmail-Labels"] = labels
    if plain is not None:
        msg.set_content(plain)
    if html is not None:
        if plain is None:
            msg.set_content(html, subtype="html")
        else:
            msg.add_alternative(html, subtype="html")
    if calendar:
        msg.add_attachment("BEGIN:VCALENDAR\nMETHOD:REPLY", subtype="calendar")
    if pdf:
        msg.add_attachment(
            b"%PDF-1.5", maintype="application", subtype="pdf", filename="cv.pdf"
        )
    return msg


def process(msg: EmailMessage) -> tuple[str, str | None]:
    """Run one message through the pipeline with default settings."""
    return process_message(msg.as_bytes(), label="Sent", min_chars=1)


@pytest.mark.parametrize(
    ("labels", "outcome"),
    [
        ("Sent", KEPT),
        ('Opened,"Category Personal",Sent', KEPT),
        ("Inbox,Important", "not_sent"),
        ("Sent,Drafts", "draft"),
    ],
)
def test_labels_decide_what_is_mine(labels: str, outcome: str) -> None:
    """Only sent, non-draft messages are kept."""
    assert process(make_message(labels=labels))[0] == outcome


def test_auto_replies_and_calendar_responses_are_skipped() -> None:
    """Auto-replies and calendar responses are not my writing."""
    auto = make_message()
    auto["Auto-Submitted"] = "auto-replied"
    assert process(auto) == ("auto_submitted", None)
    assert process(make_message(calendar=True)) == ("calendar", None)


def test_html_preferred_and_attachment_ignored() -> None:
    """HTML part wins over plain, and attachments never leak into the text."""
    msg = make_message(plain="plain version", html="<p>html version</p>", pdf=True)
    assert process(msg) == (KEPT, "html version")


def test_html_gmail_reply() -> None:
    """Gmail HTML keeps my lines, drops quote, signature and markup."""
    html = (
        '<div dir="ltr">Hi Coralie,<div><br></div><div>Sounds   good &amp; '
        "thanks.</div><div>Best,<br>Margot</div>"
        '<div class="gmail_signature">Margot | Xomnia | +31 6</div></div>'
        '<div class="gmail_quote"><div class="gmail_attr">On Wed wrote:</div>'
        "<blockquote>recruiter text</blockquote></div>"
    )
    assert clean_text(html_to_text(html)) == (
        "Hi Coralie,\n\nSounds good & thanks.\nBest,\nMargot"
    )


def test_html_outlook_reply() -> None:
    """Outlook paragraphs keep blank lines; the reply container ends my text."""
    html = (
        "<p>Will do.</p><p>&nbsp;</p><p>Margot</p>"
        '<div id="divRplyFwdMsg"><b>From:</b> Bob</div><p>old</p>'
    )
    assert clean_text(html_to_text(html)) == "Will do.\n\nMargot"


@pytest.mark.parametrize(
    "tail",
    [
        "On Wed, Oct 21, 2026 at 3:50 PM Coralie <\nc@example.com> wrote:\n> quoted",
        "Op 20 okt. 2026 om 10:00 heeft Jan <j@x.nl> het volgende\ngeschreven:\n> q",
        "-- \nMargot R\nXomnia",
        "Sent from my iPhone",
        "---------- Forwarded message ---------\nFrom: A\nhello",
        "From: Bob <b@x.com>\nSent: Monday\nSubject: x\nold text",
    ],
)
def test_clean_text_cuts_reply_signature_and_forward(tail: str) -> None:
    """Everything from a reply header, signature or forward marker is removed."""
    assert clean_text(f"Thanks, see you then!\n\n{tail}") == "Thanks, see you then!"


def test_clean_text_keeps_own_text_that_looks_like_markers() -> None:
    """Lines starting with 'On' or 'From' in my own text survive."""
    text = "On Monday I will send it.\n\nFrom my side all good.\nCheers"
    assert clean_text(text) == text


def test_clean_text_normalizes_whitespace_and_quotes() -> None:
    """Quoted lines go, nbsp and CRLF normalize, blank lines collapse."""
    assert clean_text("Yes\xa0please  \r\n> old\r\n\n\n\nBye\n") == "Yes please\n\nBye"


def test_end_to_end_mbox_to_jsonl(tmp_path: Path) -> None:
    """Mbox in, notebook-ready JSONL out, with only my cleaned sent messages."""
    mbox_path, out_path = tmp_path / "takeout.mbox", tmp_path / "out.jsonl"
    box = mailbox.mbox(mbox_path)
    box.add(make_message(plain="Great, thanks!\n\nOn Mon, Bob wrote:\n> hi", pdf=True))
    box.add(make_message(labels="Inbox", plain="Newsletter"))
    box.add(make_message(plain="---------- Forwarded message ---------\nhi"))
    box.add(make_message(plain="Sure,\nI'm in."))
    box.close()

    n = write_jsonl(extract_sent(mbox_path, "Sent", min_chars=1), out_path)

    rows = [json.loads(line) for line in out_path.read_text("utf-8").splitlines()]
    assert n == 2
    assert rows == [{"text": "Great, thanks!"}, {"text": "Sure,\nI'm in."}]
