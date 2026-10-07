from __future__ import annotations

import logging
import os
import smtplib
import ssl
from datetime import datetime
from email.message import EmailMessage

from .config import DATA_DIR

log = logging.getLogger(__name__)


def _write_outbox(subject: str, text: str, reason: str) -> str:
    box = DATA_DIR / "outbox"
    box.mkdir(parents=True, exist_ok=True)
    p = box / (datetime.now().strftime("%Y%m%d-%H%M%S") + ".txt")
    p.write_text(f"[not sent: {reason}]\nSubject: {subject}\n\n{text}")
    return str(p)


def send_email(cfg: dict, subject: str, text: str, html: str | None = None) -> tuple[bool, str]:
    """Send via SMTP. If email is disabled or no password is set, save to data/outbox instead (dry run)."""
    e = cfg["email"]
    pw = os.environ.get("EMAIL_PASSWORD", "")
    if not e.get("enabled") or not pw or not e.get("to"):
        reason = "email disabled" if not e.get("enabled") else "EMAIL_PASSWORD not set" if not pw else "no recipients"
        path = _write_outbox(subject, text, reason)
        log.warning("Email not sent (%s); saved to %s", reason, path)
        return False, f"not sent ({reason}); saved to {path}"
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = e.get("from") or e["username"]
    msg["To"] = ", ".join(e["to"])
    msg.set_content(text)
    if html:
        msg.add_alternative(html, subtype="html")
    try:
        if e["security"] == "ssl":
            with smtplib.SMTP_SSL(e["smtp_host"], int(e["smtp_port"]), context=ssl.create_default_context(),
                                  timeout=30) as s:
                s.login(e["username"], pw)
                s.send_message(msg)
        else:
            with smtplib.SMTP(e["smtp_host"], int(e["smtp_port"]), timeout=30) as s:
                s.starttls(context=ssl.create_default_context())
                s.login(e["username"], pw)
                s.send_message(msg)
    except Exception as ex:  # noqa: BLE001
        path = _write_outbox(subject, text, f"SMTP error: {ex}")
        log.error("SMTP send failed: %s", ex)
        return False, f"SMTP error: {ex} (saved to {path})"
    log.info("Email sent: %s", subject)
    return True, "sent"
