"""CLI:  python -m monitor <daily|alerts|check-sources|test-email|serve|test-rule> [--no-email]"""
from __future__ import annotations

import argparse
import logging
import sys

from . import config


def setup_logging() -> None:
    from logging.handlers import RotatingFileHandler
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    fh = RotatingFileHandler(config.DATA_DIR / "monitor.log", maxBytes=1_000_000, backupCount=3)
    sh = logging.StreamHandler()
    for h in (fh, sh):
        h.setFormatter(fmt)
        root.addHandler(h)
    logging.getLogger("yfinance").setLevel(logging.WARNING)
    logging.getLogger("apscheduler").setLevel(logging.WARNING)


def check_sources(cfg: dict) -> int:
    from . import sources
    bad = 0

    def t(name, fn, optional=False):
        nonlocal bad
        try:
            print(f"OK    {name}: {fn()}")
        except Exception as e:  # noqa: BLE001
            if optional:
                print(f"WARN  {name} (optional, not needed while the main source works): {e}")
            else:
                bad += 1
                print(f"FAIL  {name}: {e}")

    def px():
        s, src = sources.fetch_daily(cfg["symbol"], "3y")
        return f"{len(s)} closes via {src}, last {s.index[-1].date()} = {s.iloc[-1]:.2f}"

    def stq():
        s = sources._stooq_daily(cfg["symbol"])
        return f"{len(s)} closes, last {s.index[-1].date()}"

    def fred():
        s = sources.fetch_vix_fred()
        return f"{len(s)} rows, last {s.index[-1].date()} = {s.iloc[-1]}"

    def fng():
        rows, cur = sources.fetch_fng("2025-01-01")
        ds = sorted(rows)
        return f"{len(rows)} daily points {ds[0]} .. {ds[-1]}, latest score {rows[ds[-1]][0]:.1f}"

    t("prices (yfinance/Stooq)", px)
    t("prices (Stooq fallback only)", stq, optional=True)
    t("VIX (FRED)", fred)
    t("CNN Fear & Greed history", fng)
    for f in cfg["news"]["feeds"]:
        t(f"news feed: {f['name']}", lambda f=f: f"{len(sources.fetch_feed(f['name'], f['url']))} items")
    print("\nAll sources reachable." if not bad else f"\n{bad} source(s) failed - see messages above.")
    return 1 if bad else 0


def main() -> int:
    ap = argparse.ArgumentParser(prog="monitor")
    ap.add_argument("command", choices=["daily", "alerts", "check-sources", "test-email", "serve"])
    ap.add_argument("--no-email", action="store_true", help="run but do not send email")
    a = ap.parse_args()
    config.load_env()
    setup_logging()
    cfg = config.load_config()
    errs = config.validate_config(cfg)
    if errs:
        print("Config problems:\n - " + "\n - ".join(errs))
        return 2
    if a.command == "serve":
        from . import service
        service.main()
        return 0
    if a.command == "check-sources":
        return check_sources(cfg)
    if a.command == "test-email":
        from . import mailer
        ok, msg = mailer.send_email(cfg, "Nasdaq monitor test email", "If you can read this, email works.")
        print(msg)
        return 0 if ok else 1
    if a.command == "daily":
        from . import daily
        r = daily.run(cfg, send=not a.no_email)
        print(daily.render_text(r))
        return 0
    if a.command == "alerts":
        from . import alerts
        print(alerts.run(cfg, send=not a.no_email))
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
