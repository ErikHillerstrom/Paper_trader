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

Pass --trade-ticker to trade a *different* instrument for the risk leg
than the one the signal is computed from — e.g. signal off QQQ's own
(unleveraged) SMAs, but actually hold QLD (2x QQQ) while risk-on. This
lets a leveraged ETF's volatility decay be avoided during the exact
chop/drawdown periods the trend filter keeps you out of, while still
capturing extra upside during confirmed uptrends.

The crossover signal itself is always computed from the (unleveraged)
signal ticker's own SMAs — never from the traded instrument's.

Signal is read off each day's close and executed at the next trading
day's open, same convention as fear_greed_backtest.py.

Usage:
    python sma_crossover_backtest.py                          # SPY, full history, cash on death cross
    python sma_crossover_backtest.py --ticker QQQ
    python sma_crossover_backtest.py --ticker QQQ --safe-ticker TLT
    python sma_crossover_backtest.py --ticker QQQ --trade-ticker QLD
    python sma_crossover_backtest.py --fast 50 --slow 200
    python sma_crossover_backtest.py --capital 20000 --start 2015-01-01
    python sma_crossover_backtest.py --chart
"""

import sys
import json
import pandas as pd
import yfinance as yf
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from tabulate import tabulate
from colorama import Fore, Style, init

init(autoreset=True)

# ── Config ───────────────────────────────────────────────────────────────────

TICKER        = "SPY"     # SPY = S&P 500, QQQ = Nasdaq 100
SAFE_TICKER   = None       # e.g. "TLT" (20+yr treasuries), "SHY"/"BIL" (short treasuries), "GLD" (gold). None = cash.
START_CAPITAL = 20_000
SMA_FAST      = 50
SMA_SLOW      = 200

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)
SMA_TRADES_FILE  = DATA_DIR / "sma_backtest_trades.json"
SMA_SUMMARY_FILE = DATA_DIR / "sma_backtest_summary.csv"


@dataclass
class SMATrade:
    date: str
    action: str            # "buy_risk" | "buy_safe" | "buy_cash" (exit to cash, no safe ticker)
    reason: str             # "golden_cross" | "death_cross"
    asset: str               # ticker bought into ("" for cash)
    price: float
    sma_fast: float
    sma_slow: float
    cash_after: float
    shares_risk_after: float
    shares_safe_after: float
    equity_after: float


def _download(ticker: str, fetch_start: str, fetch_end: str) -> pd.DataFrame:
    df = yf.download(ticker, start=fetch_start, end=fetch_end,
                      auto_adjust=True, progress=False)
    if df.empty:
        raise RuntimeError(f"No price data returned for {ticker}")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df


def run_backtest(ticker: str = TICKER, capital: float = START_CAPITAL,
                  fast: int = SMA_FAST, slow: int = SMA_SLOW,
                  trade_ticker: str = None, safe_ticker: str = SAFE_TICKER,
                  start: str = None, end: str = None) -> dict:

    if fast >= slow:
        raise ValueError(f"--fast ({fast}) must be < --slow ({slow})")

    trade_ticker = trade_ticker or ticker

    # Need `slow` extra trading days of warmup before the requested start
    # so the slow SMA is already valid on day 1 of the test window.
    warmup_days = int(slow * 1.6) + 30
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

    # Detect crossovers: compare today's fast-vs-slow relationship to yesterday's.
    # (Using an int diff rather than boolean ~/& avoids a dtype pitfall where
    #  shift() upcasts a bool Series to object, breaking ~ on the NaN-filled slot.)
    fast_col, slow_col = f"sma{fast}", f"sma{slow}"
    above = price[fast_col] > price[slow_col]
    above_diff = above.astype(int).diff()
    golden_cross = above_diff == 1
    death_cross  = above_diff == -1

    # signal_map: trading day -> action, decided off that day's close,
    # executed at the *next* trading day's open
    signal_map = {}
    for i in range(len(trading_days) - 1):
        day, next_day = trading_days[i], trading_days[i + 1]
        if golden_cross.loc[day]:
            signal_map[next_day] = "buy_risk"
        elif death_cross.loc[day]:
            signal_map[next_day] = "buy_safe" if safe_ticker else "buy_cash"

    # Start already positioned according to day 0's regime, rather than
    # sitting idle in cash until the first crossover happens to fire.
    cash, shares_risk, shares_safe = capital, 0.0, 0.0
    day0 = trading_days[0]
    if above.loc[day0]:
        shares_risk = capital / float(trade_price.loc[day0, "Open"])
        cash = 0.0
    elif safe_ticker:
        shares_safe = capital / float(safe_price.loc[day0, "Open"])
        cash = 0.0

    trades: list[SMATrade] = []
    equity_curve = []

    for day in trading_days:
        if day in signal_map:
            action = signal_map[day]
            exec_price_trade = float(trade_price.loc[day, "Open"])
            exec_price_safe = float(safe_price.loc[day, "Open"]) if safe_ticker else None

            if action == "buy_risk" and (shares_safe > 0 or cash > 1) and shares_risk == 0:
                if shares_safe > 0:
                    cash += shares_safe * exec_price_safe
                    shares_safe = 0.0
                shares_risk = cash / exec_price_trade
                cash = 0.0
                trades.append(SMATrade(
                    day.strftime("%Y-%m-%d"), "buy_risk", "golden_cross", trade_ticker,
                    round(exec_price_trade, 2), round(float(price.loc[day, fast_col]), 2),
                    round(float(price.loc[day, slow_col]), 2),
                    round(cash, 2), round(shares_risk, 4), round(shares_safe, 4),
                    round(cash + shares_risk * exec_price_trade, 2)))

            elif action == "buy_safe" and shares_risk > 0:
                cash += shares_risk * exec_price_trade
                shares_risk = 0.0
                shares_safe = cash / exec_price_safe
                cash = 0.0
                trades.append(SMATrade(
                    day.strftime("%Y-%m-%d"), "buy_safe", "death_cross", safe_ticker,
                    round(exec_price_trade, 2), round(float(price.loc[day, fast_col]), 2),
                    round(float(price.loc[day, slow_col]), 2),
                    round(cash, 2), round(shares_risk, 4), round(shares_safe, 4),
                    round(cash + shares_safe * exec_price_safe, 2)))

            elif action == "buy_cash" and shares_risk > 0:
                cash += shares_risk * exec_price_trade
                shares_risk = 0.0
                trades.append(SMATrade(
                    day.strftime("%Y-%m-%d"), "buy_cash", "death_cross", "",
                    round(exec_price_trade, 2), round(float(price.loc[day, fast_col]), 2),
                    round(float(price.loc[day, slow_col]), 2),
                    round(cash, 2), round(shares_risk, 4), round(shares_safe, 4),
                    round(cash, 2)))

        close_signal = float(price.loc[day, "Close"])
        close_trade = float(trade_price.loc[day, "Close"])
        close_safe = float(safe_price.loc[day, "Close"]) if safe_ticker else None
        equity = cash + shares_risk * close_trade + (shares_safe * close_safe if safe_ticker else 0.0)
        state = "risk" if shares_risk > 0 else ("safe" if shares_safe > 0 else "cash")
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
    final_equity = cash + shares_risk * last_price_trade + \
                   (shares_safe * float(safe_price.loc[trading_days[-1], "Close"]) if safe_ticker else 0.0)

    bh_signal_equity = capital / float(price.loc[trading_days[0], "Open"]) * last_price_signal
    bh_trade_equity  = capital / float(trade_price.loc[trading_days[0], "Open"]) * last_price_trade

    with open(SMA_TRADES_FILE, "w", encoding="utf-8") as f:
        json.dump([asdict(t) for t in trades], f, indent=2)

    return {
        "ticker": ticker, "trade_ticker": trade_ticker, "safe_ticker": safe_ticker,
        "capital": capital, "fast": fast, "slow": slow,
        "start": trading_days[0], "end": trading_days[-1],
        "trades": trades, "equity_curve": equity_curve,
        "final_cash": cash, "final_shares_risk": shares_risk, "final_shares_safe": shares_safe,
        "final_equity": final_equity,
        "buy_and_hold_equity": bh_trade_equity,
        "buy_and_hold_signal_equity": bh_signal_equity,
    }


def print_summary(result: dict):
    trades = result["trades"]
    df = pd.DataFrame([asdict(t) for t in trades]) if trades else pd.DataFrame()
    years = (result["end"] - result["start"]).days / 365.25
    safe_ticker  = result["safe_ticker"]
    trade_ticker = result["trade_ticker"]
    leveraged    = trade_ticker != result["ticker"]

    strat_ret = (result["final_equity"] / result["capital"] - 1) * 100
    bh_ret    = (result["buy_and_hold_equity"] / result["capital"] - 1) * 100
    strat_cagr = ((result["final_equity"] / result["capital"]) ** (1 / years) - 1) * 100 if years > 0 else 0
    bh_cagr    = ((result["buy_and_hold_equity"] / result["capital"]) ** (1 / years) - 1) * 100 if years > 0 else 0

    eq = pd.DataFrame(result["equity_curve"])
    running_max = eq["equity"].cummax()
    drawdown = (eq["equity"] - running_max) / running_max * 100
    max_dd = drawdown.min() if not drawdown.empty else 0

    bh_curve = result["capital"] / eq["trade_price"].iloc[0] * eq["trade_price"]
    bh_running_max = bh_curve.cummax()
    bh_drawdown = (bh_curve - bh_running_max) / bh_running_max * 100
    bh_max_dd = bh_drawdown.min() if not bh_drawdown.empty else 0

    pct_risk = (eq["state"] == "risk").mean() * 100 if not eq.empty else 0
    pct_safe = (eq["state"] == "safe").mean() * 100 if not eq.empty else 0
    pct_cash = (eq["state"] == "cash").mean() * 100 if not eq.empty else 0

    risk_label = trade_ticker if not leveraged else f"{trade_ticker} (signal: {result['ticker']})"
    label = risk_label + (f" -> {safe_ticker} on death cross" if safe_ticker else " (cash on death cross)")
    print(f"\n{Fore.WHITE}{'='*65}")
    print(f"  SMA {result['fast']}/{result['slow']} Crossover Backtest — {label}  "
          f"({result['start'].strftime('%Y-%m-%d')} to {result['end'].strftime('%Y-%m-%d')})")
    print(f"{'='*65}{Style.RESET_ALL}")

    stats = [
        ["Starting capital",        f"${result['capital']:,.2f}"],
        ["Total trades",            len(trades)],
        ["  into risk (golden)",    int((df['action']=='buy_risk').sum()) if not df.empty else 0],
        ["  into safe (death)",     int((df['action']=='buy_safe').sum()) if not df.empty else 0],
        ["  into cash (death)",     int((df['action']=='buy_cash').sum()) if not df.empty else 0],
        [f"Time in {trade_ticker}",       f"{pct_risk:.1f}%"],
    ]
    if safe_ticker:
        stats.append([f"Time in {safe_ticker}", f"{pct_safe:.1f}%"])
    else:
        stats.append(["Time in cash", f"{pct_cash:.1f}%"])
    stats += [
        ["Final equity (strategy)", f"${result['final_equity']:,.2f}"],
        ["Strategy return",         f"{strat_ret:+.2f}%"],
        ["Strategy CAGR",           f"{strat_cagr:+.2f}%"],
        ["Strategy max drawdown",   f"{max_dd:.2f}%"],
        ["", ""],
        [f"Buy & hold {trade_ticker} equity", f"${result['buy_and_hold_equity']:,.2f}"],
        ["Buy & hold return",       f"{bh_ret:+.2f}%"],
        ["Buy & hold CAGR",         f"{bh_cagr:+.2f}%"],
        ["Buy & hold max drawdown", f"{bh_max_dd:.2f}%"],
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
        ["Strategy vs buy & hold",  f"{strat_ret - bh_ret:+.2f} pts"],
    ]
    print(tabulate(stats, tablefmt="simple", colalign=("left", "right")))

    if not df.empty:
        print(f"\n  Trade log:")
        rows = [[t["date"], t["action"], t["asset"], t["price"], t["sma_fast"], t["sma_slow"], t["equity_after"]]
                for t in df.to_dict("records")]
        print(tabulate(rows, headers=["Date", "Action", "Asset", "Exec Price",
                                       f"SMA{result['fast']} ({result['ticker']})",
                                       f"SMA{result['slow']} ({result['ticker']})", "Equity After"],
                        tablefmt="simple", floatfmt=",.2f"))
        df.to_csv(SMA_SUMMARY_FILE, index=False, encoding="utf-8")

    print(f"\n  Trades saved to {SMA_TRADES_FILE}\n")


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
    title += f" (-> {safe_ticker} on death cross)" if safe_ticker else " (cash on death cross)"
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
    sells = [t for t in result["trades"] if t.action in ("buy_safe", "buy_cash")]
    if buys:
        bdates = pd.to_datetime([t.date for t in buys])
        axes[1].scatter(bdates, [signal_price_by_date[d] for d in bdates], marker="^",
                         color="#1D9E75", s=60, zorder=5, label="Golden cross (buy)")
    if sells:
        sdates = pd.to_datetime([t.date for t in sells])
        axes[1].scatter(sdates, [signal_price_by_date[d] for d in sdates], marker="v",
                         color="#D85A30", s=60, zorder=5, label="Death cross (sell)")

    axes[1].set_ylabel("Price", color="#888")
    axes[1].legend(facecolor="#111", edgecolor="#333", labelcolor="#ccc", fontsize=9)
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))

    plt.tight_layout(pad=2)
    out = DATA_DIR / "sma_backtest_chart.png"
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
    chart = "--chart" in sys.argv

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

    result = run_backtest(ticker=ticker, capital=capital, fast=fast, slow=slow,
                           trade_ticker=trade_ticker, safe_ticker=safe_ticker,
                           start=start_date, end=end_date)
    print_summary(result)
    if chart:
        plot_results(result)
