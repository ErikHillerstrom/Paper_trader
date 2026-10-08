"""
sma_crossover_backtest.py — 50/200-day SMA golden/death cross backtest
======================================================================
Standalone from the signal-based long/short strategy (backtest_6/7) and
from the Fear & Greed contrarian strategy (fear_greed_backtest.py).
Tests a classic trend-following rule on a single index-tracking ticker
(e.g. SPY for the S&P 500, QQQ for the Nasdaq 100):

    golden cross (50-day SMA crosses above 200-day SMA) -> go long the risk leg
    death cross  (50-day SMA crosses below 200-day SMA) -> exit

On a death cross, exit goes to cash by default. Pass --safe-ticker to
instead *rotate* the full position into a defensive asset (e.g. TLT
long treasuries, or SHY/BIL/GLD) rather than sitting idle, then rotate
back to the risk leg on the next golden cross.

Pass --safe-trend to only hold the safe asset while its own close is
above its 200-day SMA, and sit in cash otherwise. This avoids holding
long bonds through a bond bear market such as 2022, when stocks and
TLT fell together.

Pass --fear-skip to apply the live "crossover + fear skip" rule from the
README: a death cross that fires during extreme fear is skipped (stay
invested) until fear fades, or until the next golden cross:

    lowest fear reading over the 10 trading days ending on the death cross D
      not extreme   -> exit (normal death cross)
      extreme fear  -> skip; exit only once fear has faded on a day after D

Fear comes from the CNN Fear & Greed Index where cached (extreme < 25,
faded >= 50), and from a VIX fallback before that (extreme = VIX above
the 75th percentile of its last 252 closes, faded = at or below the median).

Pass --trade-ticker to trade a *different* instrument for the risk leg
than the one the signal is computed from — e.g. signal off QQQ's own
(unleveraged) SMAs, but actually hold QLD (2x QQQ) while risk-on. This
lets a leveraged ETF's volatility decay be avoided during the exact
chop/drawdown periods the trend filter keeps you out of, while still
capturing extra upside during confirmed uptrends.

The crossover signal itself is always computed from the (unleveraged)
signal ticker's own SMAs — never from the traded instrument's.

Signal is read off each day's close and executed at the next trading
day's open, same convention as fear_greed_backtest.py. Each trade costs
COST_PCT of the amount traded, and cash earns the 13-week T-bill rate.

Pass --compare to run a side-by-side table of the 1x and 2x variants
(plain crossover, fear skip, fear skip + TLT, fear skip + TLT with the
trend check) over the same window. The window starts when the 2x ETF
starts trading (QLD/SSO: mid-2006).

Usage:
    python sma_crossover_backtest.py                          # SPY, full history, cash on death cross
    python sma_crossover_backtest.py --ticker QQQ
    python sma_crossover_backtest.py --ticker QQQ --safe-ticker TLT
    python sma_crossover_backtest.py --ticker QQQ --safe-ticker TLT --safe-trend
    python sma_crossover_backtest.py --ticker QQQ --trade-ticker QLD
    python sma_crossover_backtest.py --ticker QQQ --trade-ticker QLD --fear-skip --safe-ticker TLT --safe-trend
    python sma_crossover_backtest.py --ticker QQQ --compare                # 2x leg defaults to QLD (SSO for SPY)
    python sma_crossover_backtest.py --ticker QQQ --compare --trade-ticker TQQQ
    python sma_crossover_backtest.py --fast 50 --slow 200
    python sma_crossover_backtest.py --capital 20000 --start 2015-01-01
    python sma_crossover_backtest.py --chart
"""

import sys
import json
import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from tabulate import tabulate
from colorama import Fore, Style, init

from fear_greed import load_fear_greed

init(autoreset=True)

# ── Config ───────────────────────────────────────────────────────────────────

TICKER        = "SPY"     # SPY = S&P 500, QQQ = Nasdaq 100
SAFE_TICKER   = None       # e.g. "TLT" (20+yr treasuries), "SHY"/"BIL" (short treasuries), "GLD" (gold). None = cash.
START_CAPITAL = 20_000
SMA_FAST      = 50
SMA_SLOW      = 200

COST_PCT       = 0.0005    # cost per trade, as a fraction of the amount traded
CASH_RATE      = "^IRX"    # 13-week T-bill yield; cash earns this daily. None = cash earns nothing.

SAFE_TREND_SMA = 200       # --safe-trend: hold the safe asset only while it closes above this SMA

FEAR_LOOKBACK    = 10      # trading days ending on the death cross checked for extreme fear
FG_EXTREME_FEAR  = 25      # Fear & Greed below this = extreme fear
FG_FADED         = 50      # Fear & Greed at or above this = fear has faded
VIX_WINDOW       = 252     # VIX fallback: percentile window
VIX_EXTREME_PCTL = 0.75    # VIX above this percentile of its window = extreme fear

LEVERAGED_2X = {"QQQ": "QLD", "SPY": "SSO"}   # default 2x leg for --compare

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)
SMA_TRADES_FILE  = DATA_DIR / "sma_backtest_trades.json"
SMA_SUMMARY_FILE = DATA_DIR / "sma_backtest_summary.csv"
SMA_COMPARE_FILE = DATA_DIR / "sma_compare_summary.csv"


@dataclass
class SMATrade:
    date: str
    action: str            # "buy_risk" | "buy_safe" | "buy_cash" (exit to cash, or safe asset's trend turned down)
    reason: str             # "golden_cross" | "death_cross" | "fear_faded" | "safe_trend_up" | "safe_trend_down"
    asset: str               # ticker bought into ("" for cash)
    price: float
    sma_fast: float
    sma_slow: float
    cash_after: float
    shares_risk_after: float
    shares_safe_after: float
    equity_after: float


_download_cache: dict = {}


def _download(ticker: str, fetch_start: str, fetch_end: str) -> pd.DataFrame:
    key = (ticker, fetch_start, fetch_end)
    if key not in _download_cache:
        # without a start date yfinance only returns the last month, so ask for full history
        span = {"start": fetch_start} if fetch_start else {"period": "max"}
        df = yf.download(ticker, end=fetch_end, auto_adjust=True, progress=False, **span)
        if df.empty:
            raise RuntimeError(f"No price data returned for {ticker}")
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.index = pd.to_datetime(df.index).tz_localize(None)
        _download_cache[key] = df
    return _download_cache[key].copy()


def _fear_flags(index: pd.DatetimeIndex, fetch_start: str, fetch_end: str):
    """Per-day (extreme_fear, fear_faded) booleans on `index`.
    Uses CNN Fear & Greed where cached, the VIX percentile fallback elsewhere."""
    vix_start = (pd.Timestamp(fetch_start) - pd.Timedelta(days=int(VIX_WINDOW * 1.6))).strftime("%Y-%m-%d") \
                if fetch_start else None
    vix = _download("^VIX", vix_start, fetch_end)["Close"]
    extreme = vix > vix.rolling(VIX_WINDOW).quantile(VIX_EXTREME_PCTL)
    faded   = vix <= vix.rolling(VIX_WINDOW).median()
    extreme = extreme.reindex(index).ffill().fillna(False).astype(bool)
    faded   = faded.reindex(index).ffill().fillna(False).astype(bool)

    try:
        fg = load_fear_greed()["score"].sort_index()
    except FileNotFoundError:
        print("  Note: no cached Fear & Greed data — using the VIX fallback throughout.")
        return extreme, faded
    # CNN's series opens with flat 50.0 placeholder days; real readings start after them
    first_real = fg[fg != 50.0].index.min()
    fg = fg[fg.index >= first_real].reindex(index).ffill(limit=5)
    has_fg = fg.notna()
    extreme[has_fg] = fg[has_fg] < FG_EXTREME_FEAR
    faded[has_fg]   = fg[has_fg] >= FG_FADED
    return extreme, faded


def run_backtest(ticker: str = TICKER, capital: float = START_CAPITAL,
                  fast: int = SMA_FAST, slow: int = SMA_SLOW,
                  trade_ticker: str = None, safe_ticker: str = SAFE_TICKER,
                  safe_trend: bool = False, fear_skip: bool = False,
                  start: str = None, end: str = None, save: bool = True) -> dict:

    if fast >= slow:
        raise ValueError(f"--fast ({fast}) must be < --slow ({slow})")
    if safe_trend and not safe_ticker:
        raise ValueError("--safe-trend needs --safe-ticker")

    trade_ticker = trade_ticker or ticker

    # Need `slow` extra trading days of warmup before the requested start
    # so the slow SMA is already valid on day 1 of the test window.
    warmup_days = int(max(slow, SAFE_TREND_SMA) * 1.6) + 30
    fetch_start = (pd.Timestamp(start) - pd.Timedelta(days=warmup_days)).strftime("%Y-%m-%d") \
                  if start else None
    fetch_end = end

    print(f"Downloading {ticker} price history"
          f"{f' from {fetch_start}' if fetch_start else ''}"
          f"{f' to {fetch_end}' if fetch_end else ''}...")
    price = _download(ticker, fetch_start, fetch_end)

    price[f"sma{fast}"] = price["Close"].rolling(fast).mean()
    price[f"sma{slow}"] = price["Close"].rolling(slow).mean()
    price = price.dropna(subset=[f"sma{fast}", f"sma{slow}"]).copy()

    if start:
        price = price[price.index >= pd.Timestamp(start)]
    if price.empty:
        raise RuntimeError("No trading days left after applying warmup + start date")

    trade_price = price
    if trade_ticker != ticker:
        print(f"Downloading {trade_ticker} price history (risk-leg instrument)...")
        trade_price = _download(trade_ticker, fetch_start, fetch_end)

    safe_price = None
    if safe_ticker:
        print(f"Downloading {safe_ticker} price history (defensive rotation asset)...")
        safe_price = _download(safe_ticker, fetch_start, fetch_end)
        # trend check uses the safe asset's own SMA, computed before trimming to the common window
        safe_price["sma_trend"] = safe_price["Close"].rolling(SAFE_TREND_SMA).mean()

    common = price.index
    if trade_ticker != ticker:
        common = common.intersection(trade_price.index)
    if safe_ticker:
        common = common.intersection(safe_price.index)
    if common.empty:
        raise RuntimeError(f"{ticker}, {trade_ticker} and {safe_ticker} have no overlapping trading days")
    if common.min() > price.index.min():
        print(f"  Note: data availability trims the effective start to {common.min().date()}.")
    price = price.loc[common].sort_index()
    if trade_ticker != ticker:
        trade_price = trade_price.loc[common].sort_index()
    else:
        trade_price = price
    if safe_ticker:
        safe_price = safe_price.loc[common].sort_index()

    trading_days = list(price.index)

    daily_rate = pd.Series(0.0, index=price.index)
    if CASH_RATE:
        irx = _download(CASH_RATE, fetch_start, fetch_end)["Close"]
        daily_rate = (irx.reindex(price.index).ffill().fillna(0) / 100 / 252)

    # Detect crossovers: compare today's fast-vs-slow relationship to yesterday's.
    # (Using an int diff rather than boolean ~/& avoids a dtype pitfall where
    #  shift() upcasts a bool Series to object, breaking ~ on the NaN-filled slot.)
    fast_col, slow_col = f"sma{fast}", f"sma{slow}"
    above = price[fast_col] > price[slow_col]
    above_diff = above.astype(int).diff()
    golden_cross = above_diff == 1
    death_cross  = above_diff == -1

    # ── Risk regime per day (close-based): crossover, optionally with the fear skip ──
    risk_on = above.copy()
    fear_skipped_crosses = 0
    if fear_skip:
        extreme, faded = _fear_flags(price.index, fetch_start, fetch_end)
        skipped, fade_seen = False, False
        for i, day in enumerate(trading_days):
            if above.iloc[i]:
                skipped = fade_seen = False
                continue
            if death_cross.iloc[i]:
                skipped = bool(extreme.iloc[max(0, i - FEAR_LOOKBACK + 1):i + 1].any())
                fade_seen = False
                fear_skipped_crosses += skipped
            elif faded.iloc[i]:
                fade_seen = True
            risk_on.iloc[i] = skipped and not fade_seen

    # ── Where the money should sit for each day's close ──
    safe_ok = pd.Series(bool(safe_ticker), index=price.index)
    if safe_trend:
        safe_ok = (safe_price["Close"] > safe_price["sma_trend"]).fillna(False)

    def target_for(day) -> str:
        if risk_on.loc[day]:
            return "risk"
        return "safe" if safe_ok.loc[day] else "cash"

    def reason_for(day, prev_target, new_target) -> str:
        if new_target == "risk":
            return "golden_cross"
        if prev_target == "risk":
            return "death_cross" if death_cross.loc[day] else "fear_faded"
        return "safe_trend_up" if new_target == "safe" else "safe_trend_down"

    # signal_map: trading day -> (target, reason), decided off the previous day's close,
    # executed at that day's open
    signal_map = {}
    for i in range(len(trading_days) - 1):
        day, next_day = trading_days[i], trading_days[i + 1]
        prev_t = target_for(trading_days[i - 1]) if i > 0 else target_for(day)
        new_t = target_for(day)
        if new_t != prev_t:
            signal_map[next_day] = (new_t, reason_for(day, prev_t, new_t))

    # Start already positioned according to day 0's regime, rather than
    # sitting idle in cash until the first crossover happens to fire.
    cash, shares_risk, shares_safe = capital, 0.0, 0.0
    day0 = trading_days[0]
    state = target_for(day0)
    if state == "risk":
        shares_risk = capital * (1 - COST_PCT) / float(trade_price.loc[day0, "Open"])
        cash = 0.0
    elif state == "safe":
        shares_safe = capital * (1 - COST_PCT) / float(safe_price.loc[day0, "Open"])
        cash = 0.0

    trades: list[SMATrade] = []
    equity_curve = []
    action_for = {"risk": "buy_risk", "safe": "buy_safe", "cash": "buy_cash"}

    for day in trading_days:
        cash *= 1 + daily_rate.loc[day]

        if day in signal_map:
            target, reason = signal_map[day]
            exec_price_trade = float(trade_price.loc[day, "Open"])
            exec_price_safe = float(safe_price.loc[day, "Open"]) if safe_ticker else None

            # liquidate whatever is held, then buy the target
            if shares_risk > 0:
                cash += shares_risk * exec_price_trade * (1 - COST_PCT)
                shares_risk = 0.0
            if shares_safe > 0:
                cash += shares_safe * exec_price_safe * (1 - COST_PCT)
                shares_safe = 0.0
            if target == "risk":
                shares_risk = cash * (1 - COST_PCT) / exec_price_trade
                cash = 0.0
            elif target == "safe":
                shares_safe = cash * (1 - COST_PCT) / exec_price_safe
                cash = 0.0
            state = target

            asset = {"risk": trade_ticker, "safe": safe_ticker, "cash": ""}[target]
            exec_price = exec_price_safe if target == "safe" else exec_price_trade
            equity_now = cash + shares_risk * exec_price_trade + (shares_safe * exec_price_safe if safe_ticker else 0.0)
            trades.append(SMATrade(
                day.strftime("%Y-%m-%d"), action_for[target], reason, asset,
                round(exec_price, 2), round(float(price.loc[day, fast_col]), 2),
                round(float(price.loc[day, slow_col]), 2),
                round(cash, 2), round(shares_risk, 4), round(shares_safe, 4),
                round(equity_now, 2)))

        close_signal = float(price.loc[day, "Close"])
        close_trade = float(trade_price.loc[day, "Close"])
        close_safe = float(safe_price.loc[day, "Close"]) if safe_ticker else None
        equity = cash + shares_risk * close_trade + (shares_safe * close_safe if safe_ticker else 0.0)
        equity_curve.append({
            "date": day, "equity": equity,
            "price": close_signal,
            "trade_price": close_trade,
            "safe_price": close_safe,
            "sma_fast": float(price.loc[day, fast_col]),
            "sma_slow": float(price.loc[day, slow_col]),
            "state": state,
        })

    last_price_signal = float(price.loc[trading_days[-1], "Close"])
    last_price_trade  = float(trade_price.loc[trading_days[-1], "Close"])
    final_equity = equity_curve[-1]["equity"]

    bh_signal_equity = capital / float(price.loc[trading_days[0], "Open"]) * last_price_signal
    bh_trade_equity  = capital / float(trade_price.loc[trading_days[0], "Open"]) * last_price_trade

    if save:
        with open(SMA_TRADES_FILE, "w", encoding="utf-8") as f:
            json.dump([asdict(t) for t in trades], f, indent=2)

    return {
        "ticker": ticker, "trade_ticker": trade_ticker, "safe_ticker": safe_ticker,
        "safe_trend": safe_trend, "fear_skip": fear_skip,
        "fear_skipped_crosses": fear_skipped_crosses,
        "capital": capital, "fast": fast, "slow": slow,
        "start": trading_days[0], "end": trading_days[-1],
        "trades": trades, "equity_curve": equity_curve,
        "final_cash": cash, "final_shares_risk": shares_risk, "final_shares_safe": shares_safe,
        "final_equity": final_equity,
        "buy_and_hold_equity": bh_trade_equity,
        "buy_and_hold_signal_equity": bh_signal_equity,
    }


def _curve_stats(equity: pd.Series, capital: float, years: float) -> dict:
    """CAGR, max drawdown and Sharpe (daily, annualised, no risk-free deduction)."""
    ret = (equity.iloc[-1] / capital - 1) * 100
    cagr = ((equity.iloc[-1] / capital) ** (1 / years) - 1) * 100 if years > 0 else 0
    max_dd = ((equity - equity.cummax()) / equity.cummax() * 100).min()
    daily = equity.pct_change().dropna()
    sharpe = daily.mean() / daily.std() * np.sqrt(252) if daily.std() > 0 else 0
    return {"final": equity.iloc[-1], "return": ret, "cagr": cagr, "max_dd": max_dd, "sharpe": sharpe}


def _strategy_label(result: dict) -> str:
    trade_ticker = result["trade_ticker"]
    leveraged = trade_ticker != result["ticker"]
    label = trade_ticker if not leveraged else f"{trade_ticker} (signal: {result['ticker']})"
    if result["fear_skip"]:
        label += " + fear skip"
    if result["safe_ticker"]:
        label += f" -> {result['safe_ticker']}{' if trend up, else cash' if result['safe_trend'] else ''} on exit"
    else:
        label += " (cash on exit)"
    return label


def print_summary(result: dict):
    trades = result["trades"]
    df = pd.DataFrame([asdict(t) for t in trades]) if trades else pd.DataFrame()
    years = (result["end"] - result["start"]).days / 365.25
    safe_ticker  = result["safe_ticker"]
    trade_ticker = result["trade_ticker"]
    leveraged    = trade_ticker != result["ticker"]

    eq = pd.DataFrame(result["equity_curve"])
    strat = _curve_stats(eq["equity"], result["capital"], years)
    bh_curve = result["capital"] / eq["trade_price"].iloc[0] * eq["trade_price"]
    bh = _curve_stats(bh_curve, result["capital"], years)

    pct_risk = (eq["state"] == "risk").mean() * 100 if not eq.empty else 0
    pct_safe = (eq["state"] == "safe").mean() * 100 if not eq.empty else 0
    pct_cash = (eq["state"] == "cash").mean() * 100 if not eq.empty else 0

    def n_reason(r):
        return int((df["reason"] == r).sum()) if not df.empty else 0

    print(f"\n{Fore.WHITE}{'='*65}")
    print(f"  SMA {result['fast']}/{result['slow']} Crossover Backtest — {_strategy_label(result)}  "
          f"({result['start'].strftime('%Y-%m-%d')} to {result['end'].strftime('%Y-%m-%d')})")
    print(f"{'='*65}{Style.RESET_ALL}")

    stats = [
        ["Starting capital",        f"${result['capital']:,.2f}"],
        ["Total trades",            len(trades)],
        ["  into risk (golden)",    n_reason("golden_cross")],
        ["  exits on death cross",  n_reason("death_cross")],
    ]
    if result["fear_skip"]:
        stats += [
            ["  death crosses skipped (fear)", result["fear_skipped_crosses"]],
            ["  exits once fear faded",        n_reason("fear_faded")],
        ]
    if result["safe_trend"]:
        stats += [
            [f"  {safe_ticker} trend up (cash -> {safe_ticker})",   n_reason("safe_trend_up")],
            [f"  {safe_ticker} trend down ({safe_ticker} -> cash)", n_reason("safe_trend_down")],
        ]
    stats.append([f"Time in {trade_ticker}", f"{pct_risk:.1f}%"])
    if safe_ticker:
        stats.append([f"Time in {safe_ticker}", f"{pct_safe:.1f}%"])
    stats.append(["Time in cash", f"{pct_cash:.1f}%"])
    stats += [
        ["Final equity (strategy)", f"${strat['final']:,.2f}"],
        ["Strategy return",         f"{strat['return']:+.2f}%"],
        ["Strategy CAGR",           f"{strat['cagr']:+.2f}%"],
        ["Strategy max drawdown",   f"{strat['max_dd']:.2f}%"],
        ["Strategy Sharpe",         f"{strat['sharpe']:.2f}"],
        ["", ""],
        [f"Buy & hold {trade_ticker} equity", f"${bh['final']:,.2f}"],
        ["Buy & hold return",       f"{bh['return']:+.2f}%"],
        ["Buy & hold CAGR",         f"{bh['cagr']:+.2f}%"],
        ["Buy & hold max drawdown", f"{bh['max_dd']:.2f}%"],
    ]
    if leveraged:
        bh_sig_ret  = (result["buy_and_hold_signal_equity"] / result["capital"] - 1) * 100
        bh_sig_cagr = ((result["buy_and_hold_signal_equity"] / result["capital"]) ** (1 / years) - 1) * 100 if years > 0 else 0
        stats += [
            ["", ""],
            [f"Buy & hold {result['ticker']} equity (unleveraged ref)", f"${result['buy_and_hold_signal_equity']:,.2f}"],
            ["Buy & hold return",       f"{bh_sig_ret:+.2f}%"],
            ["Buy & hold CAGR",         f"{bh_sig_cagr:+.2f}%"],
        ]
    stats += [
        ["", ""],
        ["Strategy vs buy & hold",  f"{strat['return'] - bh['return']:+.2f} pts"],
    ]
    print(tabulate(stats, tablefmt="simple", colalign=("left", "right")))

    if not df.empty:
        print(f"\n  Trade log:")
        rows = [[t["date"], t["action"], t["reason"], t["asset"], t["price"], t["sma_fast"], t["sma_slow"], t["equity_after"]]
                for t in df.to_dict("records")]
        print(tabulate(rows, headers=["Date", "Action", "Reason", "Asset", "Exec Price",
                                       f"SMA{result['fast']} ({result['ticker']})",
                                       f"SMA{result['slow']} ({result['ticker']})", "Equity After"],
                        tablefmt="simple", floatfmt=",.2f"))
        df.to_csv(SMA_SUMMARY_FILE, index=False, encoding="utf-8")

    print(f"\n  Trades saved to {SMA_TRADES_FILE}\n")


def run_comparison(ticker: str, lev_ticker: str = None, safe_ticker: str = "TLT",
                    capital: float = START_CAPITAL, fast: int = SMA_FAST, slow: int = SMA_SLOW,
                    start: str = None, end: str = None):
    """Run the 1x and 2x variants over one common window and print them side by side."""
    lev_ticker = lev_ticker or LEVERAGED_2X.get(ticker)
    if not lev_ticker:
        raise ValueError(f"No default 2x ETF for {ticker} — pass --trade-ticker")

    # Align every run to the window where the 2x ETF and the safe asset both trade,
    # so the 1x and 2x rows are directly comparable.
    # The safe asset also needs SAFE_TREND_SMA days of history for its trend check.
    if not start:
        lev_first  = _download(lev_ticker, None, end).index.min()
        safe_first = _download(safe_ticker, None, end).index.min() + pd.Timedelta(days=int(SAFE_TREND_SMA * 1.6))
        start = max(lev_first, safe_first).strftime("%Y-%m-%d")

    variants = [
        ("Crossover",                     dict(fear_skip=False)),
        ("+ fear skip",                   dict(fear_skip=True)),
        (f"+ fear skip, {safe_ticker}",          dict(fear_skip=True, safe_ticker=safe_ticker)),
        (f"+ fear skip, {safe_ticker} if trend up", dict(fear_skip=True, safe_ticker=safe_ticker, safe_trend=True)),
    ]
    rows, curves = [], {}
    for lev_label, trade in (("1x", ticker), ("2x", lev_ticker)):
        for name, kw in variants:
            res = run_backtest(ticker=ticker, capital=capital, fast=fast, slow=slow,
                                trade_ticker=trade, start=start, end=end, save=False,
                                **{"safe_ticker": None, **kw})
            eq = pd.DataFrame(res["equity_curve"]).set_index("date")["equity"]
            years = (res["end"] - res["start"]).days / 365.25
            s = _curve_stats(eq, capital, years)
            rows.append([f"{lev_label} {trade}", name, s["final"], s["cagr"], s["max_dd"], s["sharpe"], len(res["trades"])])
            curves[f"{lev_label} {name}"] = eq

    for trade in (ticker, lev_ticker):
        px = _download(trade, None, end)["Close"]
        px = px[(px.index >= res["start"]) & (px.index <= res["end"])]
        years = (px.index[-1] - px.index[0]).days / 365.25
        s = _curve_stats(capital / px.iloc[0] * px, capital, years)
        rows.append([trade, "Buy & hold", s["final"], s["cagr"], s["max_dd"], s["sharpe"], 0])

    headers = ["Instrument", "Variant", "Final $", "CAGR %", "Max DD %", "Sharpe", "Trades"]
    print(f"\n{Fore.WHITE}{'='*80}")
    print(f"  {ticker} {fast}/{slow} crossover — 1x vs 2x ({lev_ticker}), ${capital:,.0f}, "
          f"{res['start'].date()} to {res['end'].date()}")
    print(f"  Costs {COST_PCT:.2%}/trade, cash earns T-bills. Signal always from {ticker}'s own SMAs.")
    print(f"{'='*80}{Style.RESET_ALL}")
    print(tabulate(rows, headers=headers, tablefmt="simple", floatfmt=(None, None, ",.0f", ".1f", ".1f", ".2f", "d")))
    pd.DataFrame(rows, columns=headers).to_csv(SMA_COMPARE_FILE, index=False, encoding="utf-8")
    print(f"\n  Comparison saved to {SMA_COMPARE_FILE}\n")
    return rows, curves


def plot_results(result: dict):
    try:
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
    except ImportError:
        print("pip install matplotlib"); return

    eq = pd.DataFrame(result["equity_curve"])
    eq["date"] = pd.to_datetime(eq["date"])
    trade_ticker = result["trade_ticker"]
    leveraged = trade_ticker != result["ticker"]
    bh_curve = result["capital"] / eq["trade_price"].iloc[0] * eq["trade_price"]
    safe_ticker = result["safe_ticker"]

    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True, facecolor="#0d0d0d",
                              gridspec_kw={"height_ratios": [1, 1]})
    for ax in axes:
        ax.set_facecolor("#111"); ax.tick_params(colors="#888"); ax.spines[:].set_color("#333")

    axes[0].plot(eq["date"], eq["equity"], color="#1D9E75", lw=1.6, label="SMA crossover strategy")
    axes[0].plot(eq["date"], bh_curve, color="#378ADD", lw=1.2, ls="--", label=f"Buy & hold {trade_ticker}")
    if leveraged:
        bh_sig_curve = result["capital"] / eq["price"].iloc[0] * eq["price"]
        axes[0].plot(eq["date"], bh_sig_curve, color="#666", lw=1.0, ls=":", label=f"Buy & hold {result['ticker']} (unleveraged)")
    axes[0].axhline(result["capital"], color="#555", lw=0.8, ls=":")

    # Shade the periods spent in the defensive/safe asset
    if safe_ticker:
        in_safe = eq["state"] == "safe"
        axes[0].fill_between(eq["date"], axes[0].get_ylim()[0], axes[0].get_ylim()[1],
                              where=in_safe, color="#E0A030", alpha=0.08,
                              transform=axes[0].get_xaxis_transform(), label=f"In {safe_ticker}")

    title = f"Equity — {result['ticker']} {result['fast']}/{result['slow']}-day SMA signal"
    title += f", traded via {trade_ticker}" if leveraged else ""
    title += " + fear skip" if result["fear_skip"] else ""
    if safe_ticker:
        title += f" (-> {safe_ticker}{' if trend up' if result['safe_trend'] else ''} on exit)"
    else:
        title += " (cash on exit)"
    axes[0].set_title(title, color="#ccc", pad=8)
    axes[0].set_ylabel("USD", color="#888")
    axes[0].legend(facecolor="#111", edgecolor="#333", labelcolor="#ccc", fontsize=9)

    axes[1].plot(eq["date"], eq["price"], color="#888", lw=1.0, label=f"{result['ticker']} price (signal)")
    axes[1].plot(eq["date"], eq["sma_fast"], color="#E0A030", lw=1.2, label=f"SMA{result['fast']}")
    axes[1].plot(eq["date"], eq["sma_slow"], color="#D85A30", lw=1.2, label=f"SMA{result['slow']}")

    # Marker y-values use the signal ticker's own price (this panel's scale),
    # not the traded instrument's price, even when trade_ticker != ticker.
    signal_price_by_date = dict(zip(eq["date"], eq["price"]))
    buys  = [t for t in result["trades"] if t.action == "buy_risk"]
    sells = [t for t in result["trades"] if t.reason in ("death_cross", "fear_faded")]
    if buys:
        bdates = pd.to_datetime([t.date for t in buys])
        axes[1].scatter(bdates, [signal_price_by_date[d] for d in bdates], marker="^",
                         color="#1D9E75", s=60, zorder=5, label="Golden cross (buy)")
    if sells:
        sdates = pd.to_datetime([t.date for t in sells])
        axes[1].scatter(sdates, [signal_price_by_date[d] for d in sdates], marker="v",
                         color="#D85A30", s=60, zorder=5, label="Exit (death cross / fear faded)")

    axes[1].set_ylabel("Price", color="#888")
    axes[1].legend(facecolor="#111", edgecolor="#333", labelcolor="#ccc", fontsize=9)
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))

    plt.tight_layout(pad=2)
    out = DATA_DIR / "sma_backtest_chart.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="#0d0d0d")
    plt.show()
    print(f"Chart saved to {out}")


def plot_comparison(curves: dict, ticker: str):
    try:
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
    except ImportError:
        print("pip install matplotlib"); return

    fig, ax = plt.subplots(figsize=(13, 7), facecolor="#0d0d0d")
    ax.set_facecolor("#111"); ax.tick_params(colors="#888"); ax.spines[:].set_color("#333")
    colors = ["#888", "#378ADD", "#E0A030", "#1D9E75"]
    for i, (name, eq) in enumerate(curves.items()):
        ax.plot(eq.index, eq.values, color=colors[i % 4], lw=1.6 if name.startswith("2x") else 1.0,
                ls="-" if name.startswith("2x") else "--", label=name)
    ax.set_yscale("log")
    ax.set_title(f"{ticker} crossover variants — 1x (dashed) vs 2x (solid), log scale", color="#ccc", pad=8)
    ax.set_ylabel("USD", color="#888")
    ax.legend(facecolor="#111", edgecolor="#333", labelcolor="#ccc", fontsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.tight_layout(pad=2)
    out = DATA_DIR / "sma_compare_chart.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="#0d0d0d")
    plt.show()
    print(f"Chart saved to {out}")


if __name__ == "__main__":
    ticker       = TICKER
    trade_ticker = None
    safe_ticker  = SAFE_TICKER
    capital      = START_CAPITAL
    fast         = SMA_FAST
    slow         = SMA_SLOW
    start_date   = None
    end_date     = None
    chart      = "--chart" in sys.argv
    fear_skip  = "--fear-skip" in sys.argv
    safe_trend = "--safe-trend" in sys.argv

    if "--ticker" in sys.argv:
        ticker = sys.argv[sys.argv.index("--ticker") + 1]
    if "--trade-ticker" in sys.argv:
        trade_ticker = sys.argv[sys.argv.index("--trade-ticker") + 1]
    if "--safe-ticker" in sys.argv:
        safe_ticker = sys.argv[sys.argv.index("--safe-ticker") + 1]
    if "--capital" in sys.argv:
        capital = float(sys.argv[sys.argv.index("--capital") + 1])
    if "--fast" in sys.argv:
        fast = int(sys.argv[sys.argv.index("--fast") + 1])
    if "--slow" in sys.argv:
        slow = int(sys.argv[sys.argv.index("--slow") + 1])
    if "--start" in sys.argv:
        start_date = sys.argv[sys.argv.index("--start") + 1]
    if "--end" in sys.argv:
        end_date = sys.argv[sys.argv.index("--end") + 1]

    if "--compare" in sys.argv:
        _, curves = run_comparison(ticker, lev_ticker=trade_ticker, safe_ticker=safe_ticker or "TLT",
                                    capital=capital, fast=fast, slow=slow, start=start_date, end=end_date)
        if chart:
            plot_comparison(curves, ticker)
        sys.exit(0)

    result = run_backtest(ticker=ticker, capital=capital, fast=fast, slow=slow,
                           trade_ticker=trade_ticker, safe_ticker=safe_ticker,
                           safe_trend=safe_trend, fear_skip=fear_skip,
                           start=start_date, end=end_date)
    print_summary(result)
    if chart:
        plot_results(result)
