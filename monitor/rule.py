"""Erik's trading-signal rule. Pure functions, no network, no news input.

Rule (target position INVESTED or CASH):
  A. Uptrend (SMA50 > SMA200)            -> INVESTED
  B. Downtrend: D = most recent death cross.
     1. lowest Fear & Greed over the 10 trading days ending on D
     2. >= 25                            -> CASH
     3. < 25 (death cross "skipped"):  Fear & Greed >= 50 on any day after D up to today -> CASH, else INVESTED
VIX fallback (only if Fear & Greed history is missing): see VixProvider.
If any needed number is missing the result is UNCERTAIN - never a guess.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

EXTREME_FEAR = 25
FADED = 50
WINDOW = 10


def fng_label(score: float) -> str:
    if score < 25:
        return "Extreme Fear"
    if score < 45:
        return "Fear"
    if score <= 55:
        return "Neutral"
    if score <= 75:
        return "Greed"
    return "Extreme Greed"


def _norm(s: pd.Series) -> pd.Series:
    s = s.dropna().sort_index().copy()
    s.index = pd.to_datetime(s.index).normalize()
    return s[~s.index.duplicated(keep="last")]


class FngProvider:
    """CNN stock-market Fear & Greed (score 0-100)."""
    name = "CNN Fear & Greed"
    kind = "fng"

    def __init__(self, series: pd.Series):
        self.s = _norm(series)

    def has_data(self) -> bool:
        return len(self.s) > 0

    def current(self, d) -> dict | None:
        v = self.s.get(pd.Timestamp(d))
        if v is None:
            return None
        return {"value": float(v), "text": f"{v:.1f} ({fng_label(v)})"}

    def window_extreme(self, dates):
        """-> (extreme_fear: bool|None, shown_value). None = data missing."""
        vals = [self.s.get(d) for d in dates]
        if not vals or any(v is None for v in vals):
            return None, None
        low = float(min(vals))
        return low < EXTREME_FEAR, low

    def faded_after(self, dates):
        """True if any day scored >= 50; None if undecidable because of gaps; else False."""
        vals = [self.s.get(d) for d in dates]
        if any(v is not None and v >= FADED for v in vals):
            return True
        if any(v is None for v in vals):
            return None
        return False

    def fade_distance(self, d) -> str:
        v = self.s.get(pd.Timestamp(d))
        if v is None:
            return "Fear & Greed value for the latest day is missing"
        return f"Fear & Greed is {max(0.0, FADED - v):.1f} points below the 50 'fear has faded' level (now {v:.1f})"


class VixProvider:
    """Fallback: VIX close vs its own last 252 closes."""
    name = "VIX fallback"
    kind = "vix"

    def __init__(self, series: pd.Series):
        self.s = _norm(series)

    def has_data(self) -> bool:
        return len(self.s) >= 252

    def _stats(self, d):
        d = pd.Timestamp(d)
        if d not in self.s.index:
            return None
        w = self.s.loc[:d].tail(252)
        if len(w) < 252:
            return None
        v = float(w.iloc[-1])
        return v, float((w < v).mean()), float(w.median())

    def current(self, d) -> dict | None:
        st = self._stats(d)
        if st is None:
            return None
        v, frac, med = st
        return {"value": v, "text": f"VIX {v:.2f}, higher than {frac*100:.0f}% of last 252 closes "
                                    f"(median {med:.2f}) - {'extreme fear' if frac > 0.75 else 'fear faded' if v <= med else 'in between'}"}

    def window_extreme(self, dates):
        flags = []
        for d in dates:
            st = self._stats(d)
            if st is None:
                return None, None
            flags.append(st[1] > 0.75)
        return any(flags), sum(flags)  # shown value = number of top-quartile days in window

    def faded_after(self, dates):
        flags = []
        for d in dates:
            st = self._stats(d)
            if st is None:
                flags.append(None)
            else:
                flags.append(st[0] <= st[2])
        if any(f is True for f in flags):
            return True
        if any(f is None for f in flags):
            return None
        return False

    def fade_distance(self, d) -> str:
        st = self._stats(d)
        if st is None:
            return "VIX history is incomplete"
        v, _, med = st
        return f"VIX {v:.2f} is {max(0.0, v - med):.2f} above its 252-day median {med:.2f} ('fear faded' needs VIX at or below it)"


def _uncertain(res: dict, reason: str, code: str) -> dict:
    res.update(target="UNCERTAIN", uncertain=True, reason=reason, reason_code=code, why=reason)
    return res


def evaluate(closes: pd.Series, fear, as_of=None) -> dict:
    """Apply the rule using closes up to and including as_of (default: last close)."""
    c = _norm(closes)
    if as_of is not None:
        c = c.loc[:pd.Timestamp(as_of)]
    res = {"uncertain": False, "reason": "", "reason_code": "", "target": None, "branch": "", "why": "",
           "watch": [], "fear_name": getattr(fear, "name", "?"), "fear_kind": getattr(fear, "kind", "?"),
           "as_of": None, "close": None, "sma50": None, "sma200": None, "gap_pct": None, "trend": None,
           "last_cross": None, "fear_now": None, "lowest10": None, "death_cross_date": None}
    if len(c) < 201:
        return _uncertain(res, f"Need at least 201 daily closes, have {len(c)}.", "price_data")

    sma50 = c.rolling(50).mean()
    sma200 = c.rolling(200).mean()
    last = c.index[-1]
    res.update(as_of=last.strftime("%Y-%m-%d"), close=float(c.iloc[-1]), sma50=float(sma50.iloc[-1]),
               sma200=float(sma200.iloc[-1]))
    res["gap_pct"] = (res["sma50"] / res["sma200"] - 1) * 100
    uptrend = res["sma50"] > res["sma200"]
    res["trend"] = "uptrend" if uptrend else "downtrend"

    state = np.sign((sma50 - sma200).dropna()).replace(0, np.nan).ffill().dropna()
    changed = state != state.shift()
    changed.iloc[0] = False
    crosses = [(d, "golden" if state[d] > 0 else "death") for d in state.index[changed]]
    if crosses:
        res["last_cross"] = {"type": crosses[-1][1], "date": crosses[-1][0].strftime("%Y-%m-%d")}
    deaths = [d for d, t in crosses if t == "death"]

    cur = fear.current(last) if fear is not None else None
    res["fear_now"] = cur["text"] if cur else None

    if uptrend:
        res.update(target="INVESTED", branch="A",
                   why="Rule A: SMA50 is above SMA200 (uptrend), so the target is INVESTED.")
        if res["gap_pct"] < 2:
            res["watch"].append("Death cross may be approaching. (Informational only: backtests showed acting early "
                                "does not help.)")
        return res

    # downtrend
    if -2 <= res["gap_pct"] < 0:
        res["watch"].append("Golden cross may be approaching.")
    if not deaths:
        return _uncertain(res, "Downtrend, but no death cross found in the loaded price history.", "price_data")
    D = deaths[-1]
    res["death_cross_date"] = D.strftime("%Y-%m-%d")
    pos = c.index.get_loc(D)
    if pos < WINDOW - 1:
        return _uncertain(res, "Not enough history before the death cross to check 10 days.", "price_data")
    window = c.index[pos - WINDOW + 1: pos + 1]
    extreme, low = fear.window_extreme(window)
    if extreme is None:
        return _uncertain(res, f"{fear.name} data is missing for one or more of the 10 trading days ending on the "
                               f"death cross ({res['death_cross_date']}).", "fear_data")
    res["lowest10"] = low
    if not extreme:
        res.update(target="CASH", branch="B2",
                   why=f"Rule B: downtrend, death cross on {res['death_cross_date']}; fear was not extreme at the "
                       f"cross (10-day {'low' if fear.kind == 'fng' else 'top-quartile days'} = {low:g}), "
                       f"so the normal exit applies: CASH.")
        return res
    after = c.index[c.index > D]
    faded = fear.faded_after(after)
    if faded is None:
        return _uncertain(res, f"Death cross on {res['death_cross_date']} was skipped (extreme fear), but "
                               f"{fear.name} data is missing for some days since, so 'fear faded' cannot be "
                               f"confirmed.", "fear_data")
    if faded:
        res.update(target="CASH", branch="B3-faded",
                   why=f"Rule B: death cross on {res['death_cross_date']} was skipped (extreme fear), but fear has "
                       f"since faded (reached the 50 level), so exit now: CASH.")
    else:
        res.update(target="INVESTED", branch="B3-skipped",
                   why=f"Rule B: death cross on {res['death_cross_date']} was skipped (extreme fear at the cross) and "
                       f"fear has not faded since, so keep holding: INVESTED.")
        res["watch"].append("Skipped death cross: " + fear.fade_distance(last) + ".")
    return res


def decide_action(prev: dict, today: dict) -> str:
    if prev["target"] == "UNCERTAIN" or today["target"] == "UNCERTAIN":
        return "UNCERTAIN - no action"
    p, t = prev["target"], today["target"]
    if p == "CASH" and t == "INVESTED":
        return "BUY at tomorrow's open"
    if p == "INVESTED" and t == "CASH":
        return "SELL at tomorrow's open"
    return "HOLD" if t == "INVESTED" else "STAY IN CASH"


def signal_word(action: str) -> str:
    return action.split(" at ")[0] if action[:3] in ("BUY", "SEL") else action


def format_block(today: dict, prev: dict, action: str, used_fallback: bool = False) -> str:
    lines = [f"SIGNAL: {signal_word(action)}"]
    if today["target"] == "UNCERTAIN" or prev["target"] == "UNCERTAIN":
        lines[0] = "SIGNAL: UNCERTAIN - no action"
        lines.append(f"Why: {today['reason'] or prev['reason']}")
        if today.get("close") is not None:
            lines.append(f"QQQ close: ${today['close']:.2f} | SMA50: ${today['sma50']:.2f} | "
                         f"SMA200: ${today['sma200']:.2f} | Gap: {today['gap_pct']:+.2f}%")
        return "\n".join(lines)
    if action.startswith(("BUY", "SELL")):
        lines[0] = f"SIGNAL: {action}"
    lines.append(f"Target position: {today['target']} (previous close: {prev['target']})")
    lines.append(f"QQQ close: ${today['close']:.2f} | SMA50: ${today['sma50']:.2f} | "
                 f"SMA200: ${today['sma200']:.2f} | Gap: {today['gap_pct']:+.2f}%   [as of {today['as_of']}]")
    low = today["lowest10"]
    if today["fear_kind"] == "fng":
        low_txt = "n/a (uptrend - not needed)" if low is None else f"{low:g}"
        lines.append(f"Fear & Greed: {today['fear_now'] or 'n/a'} | lowest of 10 days to the cross: {low_txt}")
    else:
        low_txt = "n/a (uptrend - not needed)" if low is None else f"{int(low)} top-quartile day(s) in the 10-day window"
        lines.append(f"VIX FALLBACK USED (no Fear & Greed history): {today['fear_now'] or 'n/a'} | {low_txt}")
    lc = today["last_cross"]
    lines.append(f"Last cross: {lc['type']} on {lc['date']}" if lc else "Last cross: none in loaded history")
    lines.append(f"Why: {today['why']}")
    for w in today["watch"]:
        lines.append(f"Watch: {w}")
    return "\n".join(lines)
