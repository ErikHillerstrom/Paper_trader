"""Daily report: fetch everything, store it, compute Erik's signal, email the report."""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import pandas as pd
from jinja2 import Environment, FileSystemLoader, select_autoescape

from . import db, mailer, rule, sources
from .config import ROOT

log = logging.getLogger(__name__)
_env = Environment(loader=FileSystemLoader(str(ROOT / "monitor" / "templates")),
                   autoescape=select_autoescape(["html"]))

DISCLAIMER = "Signal is the mechanical output of Erik's own rule, not personalised financial advice."


def vix_label(v: float) -> str:
    return "calm" if v < 20 else "elevated" if v <= 30 else "stressed"


# ---- data refresh --------------------------------------------------------
def refresh_data(cfg: dict, issues: list[str]) -> dict:
    info = {}
    sym = cfg["symbol"]
    try:
        s, src = sources.fetch_daily(sym, "3y")
        s, _live = sources.drop_incomplete(s)
        db.upsert_prices(sym, s)
        info["prices_source"] = src
    except Exception as e:  # noqa: BLE001
        issues.append(f"{sym} price download failed: {e}")
    try:
        v, src = sources.fetch_vix()
        db.upsert_prices("VIX", v)
        info["vix_source"] = src
    except Exception as e:  # noqa: BLE001
        issues.append(f"VIX download failed: {e}")
    try:
        start = (datetime.now(timezone.utc) - pd.Timedelta(days=900)).strftime("%Y-%m-%d")
        try:
            rows, cur = sources.fetch_fng(start)
        except Exception as e:  # noqa: BLE001
            log.warning("CNN with start date failed (%s); trying without", e)
            rows, cur = sources.fetch_fng(None)
        db.upsert_fng(rows)
        info["fng_points"] = len(rows)
    except Exception as e:  # noqa: BLE001
        issues.append(f"CNN Fear & Greed download failed (stored data used if any): {e}")
    return info


# ---- signal ----------------------------------------------------------------
def compute_signal(closes: pd.Series, fng: pd.Series, vix: pd.Series) -> dict:
    """Apply the rule for today's close and the previous trading day's close."""
    out = {"used_fallback": False, "notes": []}
    if len(closes) < 202:
        unc = {"target": "UNCERTAIN", "uncertain": True, "reason": f"Only {len(closes)} QQQ closes available "
               "(need 202+ to compare two days).", "reason_code": "price_data", "watch": [], "fear_kind": "fng",
               "fear_name": "n/a", "close": None, "last_cross": None, "why": ""}
        out.update(today=unc, prev=dict(unc), action="UNCERTAIN - no action")
        out["block"] = rule.format_block(unc, unc, out["action"])
        return out
    prev_date = closes.index[-2]
    provider = rule.FngProvider(fng)
    today, prev = rule.evaluate(closes, provider), rule.evaluate(closes, provider, as_of=prev_date)
    if "fear_data" in (today["reason_code"], prev["reason_code"]):
        vp = rule.VixProvider(vix)
        if vp.has_data():
            t2, p2 = rule.evaluate(closes, vp), rule.evaluate(closes, vp, as_of=prev_date)
            out["used_fallback"] = True
            out["notes"].append("Fear & Greed history was incomplete, so the VIX fallback was used.")
            today, prev = t2, p2
    out["today"], out["prev"] = today, prev
    out["action"] = rule.decide_action(prev, today)
    out["block"] = rule.format_block(today, prev, out["action"], out["used_fallback"])
    return out


# ---- sections --------------------------------------------------------------
def market_section(closes: pd.Series, sig: dict) -> dict:
    t = sig["today"]
    if t.get("close") is None or len(closes) < 2:
        return {"available": False}
    c = float(closes.iloc[-1])
    chg = (c / float(closes.iloc[-2]) - 1) * 100
    d50, d200 = (c / t["sma50"] - 1) * 100, (c / t["sma200"] - 1) * 100
    near = [f"close within 1% of the {n}" for n, d in (("50-day SMA", d50), ("200-day SMA", d200)) if abs(d) <= 1]
    cross = None
    p = sig["prev"]
    if p.get("last_cross") and t["last_cross"] and p["last_cross"] != t["last_cross"]:
        cross = f"NEW {t['last_cross']['type'].upper()} CROSS on {t['last_cross']['date']}"
    return {"available": True, "date": closes.index[-1].strftime("%Y-%m-%d (%A)"), "close": c, "change": chg,
            "sma50": t["sma50"], "sma200": t["sma200"], "dist50": d50, "dist200": d200, "near": near, "cross": cross}


def sentiment_section(vix: pd.Series, fng: pd.Series) -> dict:
    out = {"vix": None, "fng": None, "shift": ""}
    if len(vix) >= 2:
        v, pv = float(vix.iloc[-1]), float(vix.iloc[-2])
        out["vix"] = {"date": vix.index[-1].strftime("%Y-%m-%d"), "value": v, "change": v - pv,
                      "label": vix_label(v)}
    if len(fng) >= 2:
        s, ps = float(fng.iloc[-1]), float(fng.iloc[-2])
        wk = fng[fng.index <= fng.index[-1] - pd.Timedelta(days=7)]
        out["fng"] = {"date": fng.index[-1].strftime("%Y-%m-%d"), "score": s, "label": rule.fng_label(s),
                      "prev": ps, "week": float(wk.iloc[-1]) if len(wk) else None}
    bits = []
    if out["fng"]:
        d = out["fng"]["score"] - out["fng"]["prev"]
        bits.append(f"Fear & Greed {'up' if d > 0 else 'down' if d < 0 else 'flat'} {abs(d):.1f} pts since the prior reading")
    if out["vix"]:
        d = out["vix"]["change"]
        bits.append(f"VIX {'up' if d > 0 else 'down' if d < 0 else 'flat'} {abs(d):.2f}")
    out["shift"] = "; ".join(bits)
    return out


def news_section(cfg: dict, issues: list[str]) -> dict:
    d = cfg["daily"]
    items, iss = sources.fetch_news(cfg["news"]["feeds"], cfg["news"]["keywords"], d["news_lookback_hours"])
    issues.extend(iss)
    ny = ZoneInfo("America/New_York")
    tz = ZoneInfo(cfg["timezone"])
    now_ny = datetime.now(ny)
    # last US close: 16:00 ET of the most recent weekday (today if already past 16:00 ET and weekday)
    d0 = now_ny.date()
    close_dt = datetime(d0.year, d0.month, d0.day, 16, 0, tzinfo=ny)
    while close_dt > now_ny or close_dt.weekday() >= 5:
        close_dt -= pd.Timedelta(days=1)
    mx = d["news_max_items"]

    def fmt(it):
        return {"title": it["title"], "link": it["link"], "source": it["source"],
                "time": it["published"].astimezone(tz).strftime("%a %H:%M")}
    session = [fmt(i) for i in items if i["published"] < close_dt][:mx]
    morning = [fmt(i) for i in items if i["published"] >= close_dt][:mx]
    return {"session": session, "morning": morning, "total": len(items)}


def calendar_section(cfg: dict, tz: ZoneInfo) -> dict:
    try:
        earn = sources.fetch_earnings_today(cfg["alerts"]["members"], datetime.now(tz).date())
    except Exception as e:  # noqa: BLE001
        earn = []
        log.info("calendar failed: %s", e)
    return {"earnings": earn,
            "note": "Macro releases and Fed speakers are not fetched automatically; check an economic calendar."}


def takeaway(sig: dict, market: dict, sentiment: dict) -> str:
    bits = []
    if market.get("available"):
        bits.append(f"QQQ {market['change']:+.2f}% to ${market['close']:.2f}")
    bits.append(sig["block"].splitlines()[0].replace("SIGNAL: ", "signal: ") if sig.get("block") else "signal n/a")
    if sentiment.get("vix"):
        bits.append(f"VIX {sentiment['vix']['value']:.1f} ({sentiment['vix']['label']})")
    t = "; ".join(bits)
    return t[0].upper() + t[1:] + "."


# ---- rendering -------------------------------------------------------------
def render_text(r: dict) -> str:
    L = [f"Daily report: Nasdaq-100 - {r['date']}", r["takeaway"], "", r["signal"]["block"], ""]
    for n in r["signal"]["notes"]:
        L.append(f"Note: {n}")
    m = r["market"]
    L.append("EVENING PICTURE (last US session)")
    if m.get("available"):
        L.append(f"QQQ close {m['close']:.2f} on {m['date']}, {m['change']:+.2f}% on the day. "
                 f"SMA50 {m['sma50']:.2f} ({m['dist50']:+.2f}% away), SMA200 {m['sma200']:.2f} ({m['dist200']:+.2f}% away).")
        for x in m["near"] + ([m["cross"]] if m["cross"] else []):
            L.append(f"! {x}")
    else:
        L.append("Price data unavailable this run.")
    s = r["sentiment"]
    L += ["", "SENTIMENT"]
    L.append(f"VIX {s['vix']['value']:.2f} on {s['vix']['date']} ({s['vix']['change']:+.2f}), {s['vix']['label']}"
             if s["vix"] else "VIX: unavailable this run")
    if s["fng"]:
        f = s["fng"]
        wk = f" / week earlier {f['week']:.1f}" if f["week"] is not None else ""
        L.append(f"Fear & Greed {f['score']:.1f} ({f['label']}) on {f['date']}, previous reading {f['prev']:.1f}{wk}")
    else:
        L.append("Fear & Greed: unavailable this run")
    if s["shift"]:
        L.append(f"Shift: {s['shift']}.")
    n = r["news"]
    L += ["", "NEWS - US SESSION AND EVENING"] + ([f"- {i['title']} ({i['source']}, {i['time']}) {i['link']}" for i in n["session"]] or ["- Nothing notable found."])
    L += ["", "MORNING PICTURE (since the US close)"] + ([f"- {i['title']} ({i['source']}, {i['time']}) {i['link']}" for i in n["morning"]] or ["- Nothing notable found."])
    c = r["calendar"]
    L += ["", "TODAY'S CALENDAR"] + [f"- {e}" for e in c["earnings"]] + [c["note"]]
    if r["issues"]:
        L += ["", "DATA ISSUES"] + [f"- {i}" for i in r["issues"]]
    if r.get("web_url"):
        L += ["", f"Dashboard: {r['web_url']}"]
    L += ["", DISCLAIMER]
    return "\n".join(L)


def render_html_fragment(r: dict) -> str:
    return _env.get_template("report_fragment.html").render(r=r, disclaimer=DISCLAIMER)


# ---- main ------------------------------------------------------------------
def build_report(cfg: dict, issues: list[str]) -> dict:
    tz = ZoneInfo(cfg["timezone"])
    info = refresh_data(cfg, issues)
    closes = db.load_prices(cfg["symbol"])
    fng, vix = db.load_fng(), db.load_prices("VIX")
    sig = compute_signal(closes, fng, vix)
    if closes.empty or (pd.Timestamp.now() - closes.index[-1]).days > 5:
        issues.append("Latest stored close is more than 5 days old - data may be stale.")
    if not fng.empty and (pd.Timestamp.now() - fng.index[-1]).days > 4:
        issues.append(f"Latest Fear & Greed reading is from {fng.index[-1].date()} - may be stale.")
    market = market_section(closes, sig) if not closes.empty else {"available": False}
    sentiment = sentiment_section(vix, fng)
    news = news_section(cfg, issues)
    cal = calendar_section(cfg, tz)
    r = {"date": datetime.now(tz).strftime("%Y-%m-%d"), "generated_at": datetime.now(tz).strftime("%Y-%m-%d %H:%M %Z"),
         "signal": sig, "market": market, "sentiment": sentiment, "news": news, "calendar": cal,
         "issues": issues, "sources": info,
         "web_url": cfg["web"].get("public_url", "")}
    r["takeaway"] = takeaway(sig, market, sentiment)
    return r


def run(cfg: dict, send: bool = True) -> dict:
    db.init()
    issues: list[str] = []
    try:
        r = build_report(cfg, issues)
    except Exception as e:  # noqa: BLE001
        log.exception("daily report failed")
        if send and cfg["daily"]["send_email"]:
            mailer.send_email(cfg, "Nasdaq-100 daily report FAILED", f"The daily report crashed:\n{e!r}\n\n"
                              "Check the Logs page of the dashboard.")
        raise
    text, html = render_text(r), render_html_fragment(r)
    db.save_report(r["date"], r["signal"]["block"].splitlines()[0], r["signal"]["today"]["target"], r, html, text)
    if send and cfg["daily"]["send_email"]:
        word = r["signal"]["block"].splitlines()[0].replace("SIGNAL: ", "")
        ok, msg = mailer.send_email(cfg, f"Nasdaq-100 daily report {r['date']} - {word}", text,
                                    f"<html><body style='font-family:sans-serif;max-width:720px'>{html}</body></html>")
        r["email_status"] = msg
    return r
