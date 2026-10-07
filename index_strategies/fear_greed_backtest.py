"""
fear_greed_backtest.py — Contrarian Fear & Greed index-fund backtest
======================================================================
Standalone from the signal-based long/short strategy (backtest_6/7).
Simulates dollar allocation into a single index ETF (default SPY),
sized by CNN's Fear & Greed rating:

    extreme fear  -> buy a large % of available cash
    fear          -> buy a smaller % of available cash
    neutral       -> hold
    greed         -> hold (not extreme enough to trim)
    extreme greed -> sell a % of current holdings

By default the simulation starts fully invested (START_INVESTED = capital,
all in shares from day 1) so buys only happen once a sell has freed up
cash — pass --start-invested to hold some back as cash instead.

Signal frequency:
    daily (default) - each day's CNN rating is read from that day's close
                       and executed at the next trading day's open.
    --weekly         - each week's *average* score (Mon-Sun) is converted
                       to a rating and executed once, at the open of the
                       first trading day of the following week.

Usage:
    python fear_greed_backtest.py                          # full cached FG range, daily
    python fear_greed_backtest.py --weekly                 # weekly-average signal
    python fear_greed_backtest.py --capital 20000 --start-invested 15000
    python fear_greed_backtest.py --start 2022-01-01 --end 2025-01-01
    python fear_greed_backtest.py --ticker QQQ --chart
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

from fear_greed import load_fear_greed, fetch_fear_greed, save as save_fear_greed, FG_JSON

init(autoreset=True)

# ── Config ───────────────────────────────────────────────────────────────────

TICKER          = "SPY"
START_CAPITAL   = 20_000
START_INVESTED  = None     # None = fully invested from day 1; else $ already in shares (rest = cash)

# % of available CASH spent on a buy day
BUY_PCT_EXTREME_FEAR  = 0.35
BUY_PCT_FEAR          = 0.2

# % of current HOLDINGS (share value) sold on a sell day
SELL_PCT_GREED         = 0.0    # plain "greed" now holds instead of trimming
SELL_PCT_EXTREME_GREED = 0.05

DATA_DIR   = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)
FGB_TRADES_FILE  = DATA_DIR / "fg_backtest_trades.json"
FGB_SUMMARY_FILE = DATA_DIR / "fg_backtest_summary.csv"


@dataclass
class FGTrade:
    date: str
    action: str          # "buy" | "sell"
    rating: str
    score: float
    price: float
    amount_usd: float
    shares_delta: float
    cash_after: float
    shares_after: float
    equity_after: float


def score_to_rating(score: float) -> str:
    """CNN's own bucket boundaries, confirmed against cached (score, rating) pairs."""
    if score < 25: return "extreme fear"
    if score < 45: return "fear"
    if score < 55: return "neutral"
    if score < 75: return "greed"
    return "extreme greed"


def action_for_rating(rating: str):
    """Return ('buy'|'sell'|None, pct_of_cash_or_holdings)."""
    if rating == "extreme fear": return "buy",  BUY_PCT_EXTREME_FEAR
    if rating == "fear":         return "buy",  BUY_PCT_FEAR
    if rating == "greed" and SELL_PCT_GREED > 0: return "sell", SELL_PCT_GREED
    if rating == "extreme greed":return "sell", SELL_PCT_EXTREME_GREED
    return None, 0.0


def run_backtest(ticker: str = TICKER, capital: float = START_CAPITAL,
                  start_invested: float = START_INVESTED,
                  start: str = None, end: str = None, weekly: bool = False) -> dict:

    if not FG_JSON.exists():
        print("No cached Fear & Greed data — fetching...")
        save_fear_greed(fetch_fear_greed())
    fg = load_fear_greed()

    if start: fg = fg[fg.index >= pd.Timestamp(start)]
    if end:   fg = fg[fg.index <= pd.Timestamp(end)]

    fetch_start = (fg.index.min() - pd.Timedelta(days=5)).strftime("%Y-%m-%d")
    fetch_end   = (fg.index.max() + pd.Timedelta(days=5)).strftime("%Y-%m-%d")
    print(f"Downloading {ticker} price history {fetch_start} to {fetch_end}...")
    price = yf.download(ticker, start=fetch_start, end=fetch_end,
                         auto_adjust=True, progress=False)
    if isinstance(price.columns, pd.MultiIndex):
        price.columns = price.columns.get_level_values(0)
    price.index = pd.to_datetime(price.index).tz_localize(None)

    trading_days = [d for d in price.index if fg.index.min() <= d <= fg.index.max()]

    if start_invested is None:
        start_invested = capital
    start_invested = min(start_invested, capital)
    entry_price = float(price.loc[trading_days[0], "Open"])
    cash   = capital - start_invested
    shares = start_invested / entry_price
    trades: list[FGTrade] = []
    equity_curve = []

    # signal_map: trading day -> (rating, score) to act on, executed at that day's open
    if weekly:
        weekly_avg = fg["score"].resample("W-SUN").mean().dropna()
        signal_map = {}
        for week_end, avg_score in weekly_avg.items():
            future_days = [d for d in trading_days if d > week_end]
            if future_days:
                signal_map[future_days[0]] = (score_to_rating(avg_score), float(avg_score))
    else:
        signal_map = {}
        for i in range(len(trading_days) - 1):
            day, next_day = trading_days[i], trading_days[i + 1]
            if day in fg.index:
                signal_map[next_day] = (fg.loc[day, "rating"], float(fg.loc[day, "score"]))

    for day in trading_days:
        if day in signal_map:
            rating, score = signal_map[day]
            action, pct = action_for_rating(rating)
            exec_price = float(price.loc[day, "Open"])

            if action == "buy" and cash > 1:
                amount = cash * pct
                delta  = amount / exec_price
                cash  -= amount
                shares += delta
                trades.append(FGTrade(day.strftime("%Y-%m-%d"), "buy", rating, score,
                                       round(exec_price, 2), round(amount, 2), round(delta, 4),
                                       round(cash, 2), round(shares, 4),
                                       round(cash + shares * exec_price, 2)))
            elif action == "sell" and shares > 0:
                delta   = shares * pct
                amount  = delta * exec_price
                shares -= delta
                cash   += amount
                trades.append(FGTrade(day.strftime("%Y-%m-%d"), "sell", rating, score,
                                       round(exec_price, 2), round(amount, 2), round(-delta, 4),
                                       round(cash, 2), round(shares, 4),
                                       round(cash + shares * exec_price, 2)))

        close_px = float(price.loc[day, "Close"])
        day_score  = float(fg.loc[day, "score"])  if day in fg.index else None
        day_rating = fg.loc[day, "rating"]         if day in fg.index else None
        equity_curve.append({"date": day, "equity": cash + shares * close_px,
                              "invested": capital - cash, "price": close_px,
                              "score": day_score, "rating": day_rating})

    last_price = float(price.loc[trading_days[-1], "Close"])
    final_equity = cash + shares * last_price

    bh_first_price = float(price.loc[trading_days[0], "Open"])
    bh_shares = capital / bh_first_price
    bh_equity = bh_shares * last_price

    with open(FGB_TRADES_FILE, "w", encoding="utf-8") as f:
        json.dump([asdict(t) for t in trades], f, indent=2)

    return {
        "ticker": ticker, "capital": capital, "start_invested": start_invested, "weekly": weekly,
        "start": trading_days[0], "end": trading_days[-1],
        "trades": trades, "equity_curve": equity_curve,
        "final_cash": cash, "final_shares": shares, "final_equity": final_equity,
        "buy_and_hold_equity": bh_equity,
    }


def print_summary(result: dict):
    trades = result["trades"]
    df = pd.DataFrame([asdict(t) for t in trades]) if trades else pd.DataFrame()
    years = (result["end"] - result["start"]).days / 365.25

    strat_ret = (result["final_equity"] / result["capital"] - 1) * 100
    bh_ret    = (result["buy_and_hold_equity"] / result["capital"] - 1) * 100
    strat_cagr = ((result["final_equity"] / result["capital"]) ** (1 / years) - 1) * 100 if years > 0 else 0
    bh_cagr    = ((result["buy_and_hold_equity"] / result["capital"]) ** (1 / years) - 1) * 100 if years > 0 else 0

    eq = pd.DataFrame(result["equity_curve"])
    running_max = eq["equity"].cummax()
    drawdown = (eq["equity"] - running_max) / running_max * 100
    max_dd = drawdown.min() if not drawdown.empty else 0
    peak_invested = eq["invested"].max() if not eq.empty else result["start_invested"]

    mode = "weekly-average" if result["weekly"] else "daily"
    print(f"\n{Fore.WHITE}{'='*65}")
    print(f"  Fear & Greed Contrarian Backtest — {result['ticker']} ({mode} signal)  "
          f"({result['start'].strftime('%Y-%m-%d')} to {result['end'].strftime('%Y-%m-%d')})")
    print(f"{'='*65}{Style.RESET_ALL}")

    print(tabulate([
        ["Starting capital",        f"${result['capital']:,.2f}"],
        ["Starting invested",       f"${result['start_invested']:,.2f}"],
        ["Total trades",            len(trades)],
        ["  buys",                  int((df['action']=='buy').sum())  if not df.empty else 0],
        ["  sells",                 int((df['action']=='sell').sum()) if not df.empty else 0],
        ["Final cash",              f"${result['final_cash']:,.2f}"],
        ["Final amount invested",   f"${result['capital'] - result['final_cash']:,.2f}"],
        ["Peak amount invested",    f"${peak_invested:,.2f}"],
        ["Final shares held",       f"{result['final_shares']:.3f}"],
        ["Final equity (strategy)", f"${result['final_equity']:,.2f}"],
        ["Strategy return",         f"{strat_ret:+.2f}%"],
        ["Strategy CAGR",           f"{strat_cagr:+.2f}%"],
        ["Max drawdown",            f"{max_dd:.2f}%"],
        ["", ""],
        ["Buy & hold equity",       f"${result['buy_and_hold_equity']:,.2f}"],
        ["Buy & hold return",       f"{bh_ret:+.2f}%"],
        ["Buy & hold CAGR",         f"{bh_cagr:+.2f}%"],
        ["", ""],
        ["Strategy vs buy & hold",  f"{strat_ret - bh_ret:+.2f} pts"],
    ], tablefmt="simple", colalign=("left", "right")))

    if not df.empty:
        by_rating = df.groupby(["rating", "action"]).agg(
            events=("amount_usd", "count"), total_usd=("amount_usd", "sum")
        ).reset_index()
        print(f"\n  By rating:")
        print(tabulate(by_rating.values.tolist(),
                        headers=["Rating", "Action", "Events", "Total $"],
                        tablefmt="simple", floatfmt=",.2f"))

    df.to_csv(FGB_SUMMARY_FILE, index=False, encoding="utf-8") if not df.empty else None
    print(f"\n  Trades saved to {FGB_TRADES_FILE}\n")


def plot_results(result: dict):
    try:
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
    except ImportError:
        print("pip install matplotlib"); return

    eq = pd.DataFrame(result["equity_curve"])
    eq["date"] = pd.to_datetime(eq["date"])
    bh_curve = result["capital"] / eq["price"].iloc[0] * eq["price"]

    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True, facecolor="#0d0d0d",
                              gridspec_kw={"height_ratios": [3, 1]})
    for ax in axes:
        ax.set_facecolor("#111"); ax.tick_params(colors="#888"); ax.spines[:].set_color("#333")

    axes[0].plot(eq["date"], eq["equity"], color="#1D9E75", lw=1.6, label="Fear & Greed strategy")
    axes[0].plot(eq["date"], bh_curve, color="#378ADD", lw=1.2, ls="--", label="Buy & hold")
    axes[0].plot(eq["date"], eq["invested"], color="#E0A030", lw=1.2, ls=":",
                 label="Amount invested (cost basis)")
    axes[0].axhline(result["capital"], color="#555", lw=0.8, ls=":")
    axes[0].set_title(f"Equity — {result['ticker']} Fear & Greed contrarian strategy",
                       color="#ccc", pad=8)
    axes[0].set_ylabel("USD", color="#888")
    axes[0].legend(facecolor="#111", edgecolor="#333", labelcolor="#ccc", fontsize=9)

    colors = {"extreme fear": "#1D9E75", "fear": "#7FBF7F", "neutral": "#888",
              "greed": "#E0A030", "extreme greed": "#D85A30"}
    axes[1].scatter(eq["date"], eq["score"], s=4,
                     c=[colors.get(r, "#888") for r in eq["rating"]])
    axes[1].set_ylabel("F&G score", color="#888")
    axes[1].axhline(25, color="#555", lw=0.6, ls="--")
    axes[1].axhline(75, color="#555", lw=0.6, ls="--")
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))

    plt.tight_layout(pad=2)
    out = DATA_DIR / "fg_backtest_chart.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="#0d0d0d")
    plt.show()
    print(f"Chart saved to {out}")


if __name__ == "__main__":
    ticker  = TICKER
    capital = START_CAPITAL
    start_invested = START_INVESTED
    start_date = None
    end_date   = None
    chart  = "--chart"  in sys.argv
    weekly = "--weekly" in sys.argv

    if "--ticker" in sys.argv:
        ticker = sys.argv[sys.argv.index("--ticker") + 1]
    if "--capital" in sys.argv:
        capital = float(sys.argv[sys.argv.index("--capital") + 1])
    if "--start-invested" in sys.argv:
        start_invested = float(sys.argv[sys.argv.index("--start-invested") + 1])
    if "--start" in sys.argv:
        start_date = sys.argv[sys.argv.index("--start") + 1]
    if "--end" in sys.argv:
        end_date = sys.argv[sys.argv.index("--end") + 1]

    result = run_backtest(ticker=ticker, capital=capital, start_invested=start_invested,
                           start=start_date, end=end_date, weekly=weekly)
    print_summary(result)
    if chart:
        plot_results(result)
