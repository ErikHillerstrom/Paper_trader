"""Big-moves alert scan. Prices via yfinance (delayed is fine); Fear & Greed is read from the local database."""
from __future__ import annotations

import logging

import pandas as pd

from . import daily, db, mailer, rule, sources

log = logging.getLogger(__name__)


def _members_moves(members: list[str]) -> dict[str, float]:
    import yfinance as yf
    df = yf.download(members, period="7d", interval="1d", auto_adjust=False, progress=False)["Close"]
    out = {}
    for m in members:
        col = df[m].dropna() if m in df else pd.Series(dtype=float)
        if len(col) >= 2:
            out[m] = (float(col.iloc[-1]) / float(col.iloc[-2]) - 1) * 100
    return out


def collect(cfg: dict) -> tuple[list[tuple], list[str]]:
    """Returns ([(key, kind, subject, body)], issues)."""
    th = cfg["alerts"]["thresholds"]
    sym = cfg["symbol"]
    found: list[tuple] = []
    issues: list[str] = []

    # --- fresh prices (completed rows are stored, the forming session row is only used as the live price)
    live = prev_close = None
    try:
        s, _ = sources.fetch_daily(sym, "1y")
        done, _ = sources.drop_incomplete(s)
        db.upsert_prices(sym, done)
        live_date, live = s.index[-1], float(s.iloc[-1])
        prev_close = float(s.iloc[-2])
    except Exception as e:  # noqa: BLE001
        issues.append(f"{sym} live price failed: {e}")

    closes = db.load_prices(sym)
    fng, vix_hist = db.load_fng(), db.load_prices("VIX")

    if live is not None:
        d = live_date.strftime("%Y-%m-%d")
        chg = (live / prev_close - 1) * 100
        tier = int(abs(chg) // th["qqq_move_pct"]) if th["qqq_move_pct"] > 0 else 0
        if tier >= 1:
            way = "up" if chg > 0 else "down"
            found.append((f"qqq_move:{d}:{way}:{tier}", "move",
                          f"QQQ {chg:+.2f}% ({d})",
                          f"QQQ is at ${live:.2f}, {chg:+.2f}% versus the previous close of ${prev_close:.2f} "
                          f"(threshold {th['qqq_move_pct']}%)."))
        # proximity to SMAs (SMAs from stored completed closes)
        if len(closes) >= 200:
            for n in (50, 200):
                sma = float(closes.rolling(n).mean().iloc[-1])
                dist = (live / sma - 1) * 100
                if abs(dist) <= th["sma_proximity_pct"]:
                    found.append((f"sma{n}:{d}", "sma", f"QQQ within {abs(dist):.2f}% of its {n}-day SMA",
                                  f"QQQ ${live:.2f} vs SMA{n} ${sma:.2f} ({dist:+.2f}%)."))

    # --- VIX
    try:
        v, _ = sources.fetch_daily("^VIX", "1mo")
        vv, pv, vd = float(v.iloc[-1]), float(v.iloc[-2]), v.index[-1].strftime("%Y-%m-%d")
        spike = (vv / pv - 1) * 100
        if spike >= th["vix_spike_pct"]:
            found.append((f"vix_spike:{vd}", "vix", f"VIX jumped {spike:+.1f}% to {vv:.1f}",
                          f"VIX {vv:.2f} versus {pv:.2f} previous close ({vd})."))
        if vv >= th["vix_above"]:
            found.append((f"vix_level:{vd}", "vix", f"VIX at {vv:.1f} (above {th['vix_above']:g})",
                          f"VIX {vv:.2f} on {vd}; threshold {th['vix_above']:g}."))
    except Exception as e:  # noqa: BLE001
        issues.append(f"VIX live check failed: {e}")

    # --- Fear & Greed from the database
    if len(fng) >= 2:
        s, ps, fd = float(fng.iloc[-1]), float(fng.iloc[-2]), fng.index[-1].strftime("%Y-%m-%d")
        if abs(s - ps) >= th["fng_change_points"]:
            found.append((f"fng_change:{fd}", "fng", f"Fear & Greed moved {s - ps:+.1f} to {s:.1f}",
                          f"{s:.1f} ({rule.fng_label(s)}) on {fd}, previous stored reading {ps:.1f}."))
        if s < th["fng_extreme_low"] <= ps:
            found.append((f"fng_low:{fd}", "fng", f"Fear & Greed entered extreme fear ({s:.1f})",
                          f"Score {s:.1f} on {fd}, previous {ps:.1f}."))
        if s > th["fng_extreme_high"] >= ps:
            found.append((f"fng_high:{fd}", "fng", f"Fear & Greed entered extreme greed ({s:.1f})",
                          f"Score {s:.1f} on {fd}, previous {ps:.1f}."))
        if (pd.Timestamp.now() - fng.index[-1]).days > 4:
            found.append((f"fng_stale:{pd.Timestamp.now():%Y-%m-%d}", "health",
                          "Stored Fear & Greed data is stale",
                          f"Latest stored reading is from {fd}. The daily report may be failing to fetch it."))

    # --- signal change on completed closes (stored data only)
    try:
        if len(closes) >= 202:
            sig = daily.compute_signal(closes, fng, vix_hist)
            a = sig["action"]
            if a.startswith(("BUY", "SELL")):
                found.append((f"signal:{sig['today']['as_of']}:{a[:3]}", "signal", f"SIGNAL: {a}", sig["block"]))
    except Exception as e:  # noqa: BLE001
        issues.append(f"Signal check failed: {e}")

    # --- big members
    try:
        for m, chg in _members_moves(cfg["alerts"]["members"]).items():
            if abs(chg) >= th["member_move_pct"]:
                d = pd.Timestamp.now().strftime("%Y-%m-%d")
                found.append((f"member:{m}:{d}:{'up' if chg > 0 else 'down'}", "member",
                              f"{m} {chg:+.1f}%", f"{m} moved {chg:+.2f}% versus its previous close."))
    except Exception as e:  # noqa: BLE001
        issues.append(f"Member check failed: {e}")
    return found, issues


def run(cfg: dict, send: bool = True) -> dict:
    db.init()
    found, issues = collect(cfg)
    new = [a for a in found if db.add_alert(*a)]
    result = {"checked": len(found), "new": [a[2] for a in new], "issues": issues, "email": None}
    if new and send:
        body = "\n\n".join(f"* {a[2]}\n{a[3]}" for a in new)
        if cfg["web"].get("public_url"):
            body += f"\n\nDashboard: {cfg['web']['public_url']}"
        subj = new[0][2] if len(new) == 1 else f"{len(new)} market alerts: {new[0][2]} (+{len(new) - 1} more)"
        ok, msg = mailer.send_email(cfg, f"[Nasdaq alert] {subj}", body)
        result["email"] = msg
        if ok:
            db.mark_alerts_emailed([a[0] for a in new])
    for i in issues:
        log.warning("alert scan issue: %s", i)
    return result
