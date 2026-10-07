"""All network data access. Every function raises on failure; callers decide how to degrade."""
from __future__ import annotations

import io
import logging
import time
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from zoneinfo import ZoneInfo

import pandas as pd
import requests
from defusedxml import ElementTree as ET

log = logging.getLogger(__name__)
UA = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
      "Chrome/124.0 Safari/537.36")
NY = ZoneInfo("America/New_York")


def http_get(url: str, headers: dict | None = None, tries: int = 3, timeout: int = 30) -> requests.Response:
    last = None
    for i in range(tries):
        try:
            r = requests.get(url, headers={"User-Agent": UA, **(headers or {})}, timeout=timeout)
            r.raise_for_status()
            return r
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(2 * (i + 1))
    raise RuntimeError(f"GET {url} failed: {last}")


# ---- prices -------------------------------------------------------------
def _yahoo_daily(symbol: str, period: str) -> pd.Series:
    import yfinance as yf
    df = yf.Ticker(symbol).history(period=period, interval="1d", auto_adjust=False)
    if df is None or df.empty:
        raise RuntimeError(f"yfinance returned no data for {symbol}")
    s = df["Close"].dropna()
    s.index = pd.DatetimeIndex(s.index).tz_localize(None).normalize()
    return s.astype(float)


def _stooq_daily(symbol: str) -> pd.Series:
    code = symbol.lower() if symbol.startswith("^") else symbol.lower() + ".us"
    r = http_get(f"https://stooq.com/q/d/l/?s={code}&i=d")
    df = pd.read_csv(io.StringIO(r.text))
    if "Close" not in df.columns or df.empty:
        raise RuntimeError(f"stooq returned no usable data for {symbol}: {r.text[:80]!r}")
    s = pd.Series(df["Close"].astype(float).values, index=pd.to_datetime(df["Date"]))
    return s.sort_index()


def fetch_daily(symbol: str, period: str = "3y") -> tuple[pd.Series, str]:
    """Daily closes (unadjusted). Yahoo first, Stooq as fallback. Returns (series, source_name)."""
    try:
        return _yahoo_daily(symbol, period), "yfinance"
    except Exception as e:  # noqa: BLE001
        log.warning("yfinance failed for %s (%s); trying Stooq", symbol, e)
    s = _stooq_daily(symbol)
    years = int(period.rstrip("y")) if period.endswith("y") else 1
    return s[s.index >= s.index[-1] - pd.Timedelta(days=365 * years)], "stooq"


def drop_incomplete(s: pd.Series, now: datetime | None = None) -> tuple[pd.Series, tuple | None]:
    """Remove a still-forming US session row. Returns (completed_closes, (date, last_price) | None for live row)."""
    if s.empty:
        return s, None
    now_ny = (now or datetime.now(timezone.utc)).astimezone(NY)
    last_date = s.index[-1].date()
    session_open = last_date == now_ny.date() and (now_ny.hour, now_ny.minute) < (16, 15)
    if session_open:
        return s.iloc[:-1], (s.index[-1], float(s.iloc[-1]))
    return s, None


def fetch_vix_fred() -> pd.Series:
    # FRED stalls requests that send a browser-style User-Agent, but answers plain clients instantly.
    r = http_get("https://fred.stlouisfed.org/graph/fredgraph.csv?id=VIXCLS",
                 headers={"User-Agent": "curl/8.5.0"}, tries=2, timeout=15)
    df = pd.read_csv(io.StringIO(r.text))
    dates = pd.to_datetime(df.iloc[:, 0])
    vals = pd.to_numeric(df.iloc[:, 1], errors="coerce")
    s = pd.Series(vals.values, index=dates).dropna()
    if len(s) < 100:
        raise RuntimeError("FRED VIXCLS returned too few rows")
    return s


def fetch_vix() -> tuple[pd.Series, str]:
    try:
        return fetch_vix_fred(), "FRED VIXCLS"
    except Exception as e:  # noqa: BLE001
        log.warning("FRED VIX failed (%s); trying Yahoo ^VIX", e)
    return _yahoo_daily("^VIX", "2y"), "yfinance ^VIX"


# ---- CNN Fear & Greed ---------------------------------------------------
def fetch_fng(start: str | None = None) -> tuple[dict, dict | None]:
    """Returns ({'YYYY-MM-DD': (score, rating)}, current_block_or_None). CNN stock-market index only."""
    base = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata"
    url = f"{base}/{start}" if start else base
    r = http_get(url, headers={"Origin": "https://edition.cnn.com", "Referer": "https://edition.cnn.com/",
                               "Accept": "application/json"})
    data = r.json()
    hist = data["fear_and_greed_historical"]["data"]
    out: dict = {}
    for p in hist:
        d = datetime.fromtimestamp(p["x"] / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
        out[d] = (float(p["y"]), p.get("rating", ""))
    cur = data.get("fear_and_greed")
    if cur and cur.get("timestamp") and cur.get("score") is not None:
        ts = cur["timestamp"]
        d = (datetime.fromtimestamp(ts / 1000, tz=timezone.utc) if isinstance(ts, (int, float))
             else datetime.fromisoformat(str(ts).replace("Z", "+00:00"))).strftime("%Y-%m-%d")
        out[d] = (float(cur["score"]), cur.get("rating", ""))
    if len(out) < 30:
        raise RuntimeError(f"CNN feed returned only {len(out)} data points")
    return out, cur


# ---- news ---------------------------------------------------------------
def _parse_time(txt: str | None) -> datetime | None:
    if not txt:
        return None
    try:
        dt = parsedate_to_datetime(txt)
    except Exception:  # noqa: BLE001
        try:
            dt = datetime.fromisoformat(txt.replace("Z", "+00:00"))
        except Exception:  # noqa: BLE001
            return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def fetch_feed(name: str, url: str) -> list[dict]:
    root = ET.fromstring(http_get(url, tries=2).content)
    items = []
    for el in root.iter():
        if _local(el.tag) not in ("item", "entry"):
            continue
        f = {_local(ch.tag): (ch.text or "").strip() for ch in el}
        link = f.get("link", "")
        if not link:
            for ch in el:
                if _local(ch.tag) == "link" and ch.get("href"):
                    link = ch.get("href")
        t = _parse_time(f.get("pubDate") or f.get("published") or f.get("updated") or f.get("date"))
        if f.get("title"):
            items.append({"title": f["title"], "link": link, "source": name, "published": t})
    return items


def fetch_news(feeds: list[dict], keywords: list[str], lookback_hours: int,
               now: datetime | None = None) -> tuple[list[dict], list[str]]:
    now = now or datetime.now(timezone.utc)
    cutoff = now - timedelta(hours=lookback_hours)
    seen, items, issues = set(), [], []
    kws = [k.lower() for k in keywords]
    for f in feeds:
        try:
            for it in fetch_feed(f.get("name", f["url"]), f["url"]):
                if it["published"] is None or it["published"] < cutoff:
                    continue
                key = it["title"].lower()[:90]
                if key in seen:
                    continue
                seen.add(key)
                it["hits"] = sum(1 for k in kws if k in it["title"].lower())
                items.append(it)
        except Exception as e:  # noqa: BLE001
            issues.append(f"News feed '{f.get('name')}' failed: {e}")
    items.sort(key=lambda i: (i["hits"], i["published"]), reverse=True)
    return items, issues


def fetch_earnings_today(tickers: list[str], today) -> list[str]:
    """Large members reporting earnings today or tomorrow (from Yahoo calendar data)."""
    import yfinance as yf
    out = []
    for t in tickers:
        try:
            cal = yf.Ticker(t).calendar
            dates = cal.get("Earnings Date", []) if isinstance(cal, dict) else []
            for d in dates:
                d = pd.Timestamp(d).date()
                if d == today:
                    out.append(f"{t} reports earnings today")
                elif d == today + timedelta(days=1):
                    out.append(f"{t} reports earnings tomorrow")
        except Exception as e:  # noqa: BLE001
            log.info("earnings lookup failed for %s: %s", t, e)
    return out
