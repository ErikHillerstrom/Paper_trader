from __future__ import annotations

import copy
import os
import shutil
from pathlib import Path
from zoneinfo import ZoneInfo

import yaml

ROOT = Path(os.environ.get("MONITOR_HOME", Path(__file__).resolve().parent.parent))
DATA_DIR = Path(os.environ.get("MONITOR_DATA", ROOT / "data"))
CONFIG_PATH = ROOT / "config.yaml"
ENV_PATH = ROOT / ".env"

DAY_NAMES = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]

DEFAULTS = {
    "timezone": "Europe/Stockholm",
    "symbol": "QQQ",
    "email": {"enabled": False, "smtp_host": "smtp.gmail.com", "smtp_port": 465, "security": "ssl",
              "username": "", "from": "", "to": []},
    "daily": {"time": "08:30", "days": "mon-fri", "catch_up": True, "send_email": True,
              "news_lookback_hours": 20, "news_max_items": 8},
    "alerts": {"enabled": True, "days": "mon-fri", "every_hours": 3, "start_hour": 9, "end_hour": 23, "minute": 5,
               "thresholds": {"qqq_move_pct": 2.0, "vix_spike_pct": 15.0, "vix_above": 25.0,
                              "sma_proximity_pct": 1.0, "fng_change_points": 10.0, "fng_extreme_low": 25.0,
                              "fng_extreme_high": 75.0, "member_move_pct": 5.0},
               "members": ["NVDA", "MSFT", "AAPL", "AMZN", "META", "GOOGL", "AVGO", "TSLA"]},
    "news": {"keywords": [], "feeds": []},
    "web": {"host": "127.0.0.1", "port": 8080, "username": "admin", "public_url": ""},
}


def load_env() -> None:
    """Load KEY=VALUE lines from .env into os.environ (existing variables win)."""
    if not ENV_PATH.exists():
        return
    for line in ENV_PATH.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config() -> dict:
    raw = {}
    if CONFIG_PATH.exists():
        raw = yaml.safe_load(CONFIG_PATH.read_text()) or {}
    return deep_merge(DEFAULTS, raw)


def parse_days(spec: str) -> list[int]:
    """'mon-fri' / 'mon,wed,fri' -> list of weekday ints (Mon=0). Raises ValueError."""
    days: set[int] = set()
    for part in str(spec).lower().replace(" ", "").split(","):
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            ia, ib = DAY_NAMES.index(a), DAY_NAMES.index(b)
            if ia > ib:
                raise ValueError(f"bad day range '{part}'")
            days.update(range(ia, ib + 1))
        else:
            days.add(DAY_NAMES.index(part))
    if not days:
        raise ValueError("no days selected")
    return sorted(days)


def days_for_cron(spec: str) -> str:
    return ",".join(DAY_NAMES[i] for i in parse_days(spec))


def validate_config(cfg: dict) -> list[str]:
    errs: list[str] = []

    def num(path, val, lo=None, hi=None):
        if not isinstance(val, (int, float)) or isinstance(val, bool):
            errs.append(f"{path} must be a number")
        elif (lo is not None and val < lo) or (hi is not None and val > hi):
            errs.append(f"{path} must be between {lo} and {hi}")

    try:
        ZoneInfo(cfg["timezone"])
    except Exception:
        errs.append("timezone is not a valid IANA timezone (e.g. Europe/Stockholm)")
    t = str(cfg["daily"]["time"])
    try:
        h, m = t.split(":")
        assert 0 <= int(h) <= 23 and 0 <= int(m) <= 59
    except Exception:
        errs.append("daily.time must be HH:MM")
    for sect in ("daily", "alerts"):
        try:
            parse_days(cfg[sect]["days"])
        except Exception:
            errs.append(f"{sect}.days must look like mon-fri or mon,wed,fri")
    a = cfg["alerts"]
    num("alerts.every_hours", a["every_hours"], 1, 12)
    num("alerts.start_hour", a["start_hour"], 0, 23)
    num("alerts.end_hour", a["end_hour"], 0, 23)
    num("alerts.minute", a["minute"], 0, 59)
    if isinstance(a["start_hour"], (int, float)) and isinstance(a["end_hour"], (int, float)) \
            and a["start_hour"] > a["end_hour"]:
        errs.append("alerts.start_hour must not be after alerts.end_hour")
    for k, v in a["thresholds"].items():
        num(f"alerts.thresholds.{k}", v, 0, 1000)
    if not isinstance(cfg["email"]["to"], list) or not all(isinstance(x, str) and "@" in x for x in cfg["email"]["to"]):
        errs.append("email.to must be a list of email addresses")
    if cfg["email"]["security"] not in ("ssl", "starttls"):
        errs.append("email.security must be ssl or starttls")
    num("daily.news_lookback_hours", cfg["daily"]["news_lookback_hours"], 1, 72)
    num("daily.news_max_items", cfg["daily"]["news_max_items"], 1, 30)
    num("web.port", cfg["web"]["port"], 1, 65535)
    for f in cfg["news"]["feeds"]:
        if not isinstance(f, dict) or "url" not in f or not str(f["url"]).startswith("http"):
            errs.append("each news feed needs a name and an http(s) url")
            break
    return errs


def save_config(cfg: dict) -> list[str]:
    errs = validate_config(cfg)
    if errs:
        return errs
    if CONFIG_PATH.exists():
        shutil.copy2(CONFIG_PATH, CONFIG_PATH.with_suffix(".yaml.bak"))
    tmp = CONFIG_PATH.with_suffix(".yaml.tmp")
    tmp.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
    tmp.replace(CONFIG_PATH)
    return []
