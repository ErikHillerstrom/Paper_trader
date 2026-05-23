"""
signal_path_analysis.py — Average 5-min price path by signal type
==================================================================
For every signal triggered over 5 years, records the normalized price
path over the following 3 trading days using Alpaca IEX 5-min bars.
Groups by signal name to reveal characteristic movement patterns
(e.g. does gap_up peak in the first hour or later?).

All paths are normalized: entry price = 0%, shorts inverted so that
"up" always means "in direction of the signal". This lets you compare
long and short signals on the same chart.

Usage:
    python signal_path_analysis.py              # 5 years
    python signal_path_analysis.py --days 500   # shorter window
    python signal_path_analysis.py --no-cache   # force re-download
"""

import os
import sys
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path
from datetime import datetime, timedelta

# ── Config ─────────────────────────────────────────────────────────────────────

ALPACA_API_KEY    = os.environ.get("ALPACA_API_KEY",    "")
ALPACA_API_SECRET = os.environ.get("ALPACA_API_SECRET", "")

DAYS_BACK          = 1825    # calendar days to scan (≈5 years)
BARS_PER_DAY       = 78      # 9:30–16:00 ET = 390 min / 5 = 78 bars
HOLD_DAYS          = 3       # how many trading days to record after entry
BARS_AHEAD         = BARS_PER_DAY * HOLD_DAYS
MIN_TRADES         = 8       # minimum signal occurrences to plot a group

CALL_PUT_RATIO_MIN = 3.0
VPIN_THRESHOLD     = 0.65
LONG_MIN_SIGNALS   = 2
SHORT_MIN_SIGNALS  = 2
LONG_SCORE_MIN     = 0.6
SHORT_SCORE_MIN    = 0.6

WATCHLIST = [
    "NVDA","MSFT","AAPL","AMZN","META","GOOGL","TSLA","JPM",
    "XOM","PFE","MRNA","AMD","NFLX","CRM","INTC","BAC","GS",
    "ABBV","LLY","UNH","V","MA","AVGO","ORCL","ADBE",
]

DATA_DIR  = Path(__file__).parent / "data"
CACHE_DIR = DATA_DIR / "cache"
DATA_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)

import pickle

def _save(obj, path):
    with open(path, "wb") as f: pickle.dump(obj, f)

def _load(path):
    with open(path, "rb") as f: return pickle.load(f)

# ── Data loading ───────────────────────────────────────────────────────────────

def load_daily(days_back: int, force: bool = False) -> dict:
    cache = CACHE_DIR / f"daily_{days_back}d.pkl"
    if cache.exists() and not force:
        print("Loading daily bars from cache...")
        history = _load(cache)
        print(f"  {len(history)} tickers loaded from cache\n")
        return history

    end   = datetime.now()
    start = end - timedelta(days=days_back + 90)
    print(f"Downloading {days_back}d daily bars for {len(WATCHLIST)} tickers...")
    raw = yf.download(
        WATCHLIST, start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"),
        interval="1d", group_by="ticker", auto_adjust=True, progress=False,
    )
    history = {}
    for ticker in WATCHLIST:
        try:
            df = raw[ticker].dropna(how="all").copy() if len(WATCHLIST) > 1 \
                 else raw.dropna(how="all").copy()
            df.index = pd.to_datetime(df.index).tz_localize(None)
            history[ticker] = df
        except Exception as e:
            print(f"  {ticker}: {e}")

    _save(history, cache)
    print(f"  {len(history)} tickers loaded\n")
    return history


def load_intraday(days_back: int, force: bool = False) -> dict:
    cache = CACHE_DIR / f"alpaca_5min_{days_back}d.pkl"
    if cache.exists() and not force:
        print("Loading 5-min bars from cache...")
        history = _load(cache)
        print(f"  {len(history)} tickers loaded from cache\n")
        return history

    try:
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
    except ImportError:
        print("ERROR: run  pip install alpaca-py"); sys.exit(1)

    end   = datetime.now()
    start = end - timedelta(days=days_back)
    print(f"Downloading 5-min Alpaca IEX bars ({start.date()} to {end.date()}) "
          f"for {len(WATCHLIST)} tickers...")
    print("  (this may take several minutes for 5 years of data...)")

    client  = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_API_SECRET)
    request = StockBarsRequest(
        symbol_or_symbols=WATCHLIST,
        timeframe=TimeFrame(5, TimeFrameUnit.Minute),
        start=start, end=end, feed="iex", adjustment="split", limit=None,
    )
    bars = client.get_stock_bars(request)
    raw  = bars.df

    history = {}
    for ticker in WATCHLIST:
        try:
            if ticker not in raw.index.get_level_values("symbol"):
                continue
            df = raw.xs(ticker, level="symbol").copy()
            df.index = pd.to_datetime(df.index).tz_localize(None)
            df.rename(columns={"open":"Open","high":"High","low":"Low",
                                "close":"Close","volume":"Volume"}, inplace=True)
            history[ticker] = df
        except Exception as e:
            print(f"  {ticker}: {e}")

    _save(history, cache)
    print(f"  {len(history)} tickers loaded\n")
    return history

# ── Signal detection (REQUIRE_DIRECTIONAL=False to capture all combos) ────────

def compute_signals(date: datetime, hist: pd.DataFrame):
    """Returns (long_signals, long_score, short_signals, short_score) or None."""
    past = hist[hist.index <= pd.Timestamp(date)].copy()
    if len(past) < 25:
        return None

    today   = past.iloc[-1]
    prior   = past.iloc[-2]
    avg_vol = past["Volume"].iloc[-21:-1].mean()
    if avg_vol == 0:
        return None

    vol_spike = bool(today["Volume"] > avg_vol * 2.0)
    block     = bool(today["Volume"] > avg_vol * 1.5)
    gap_up    = bool(today["Open"]   > prior["Close"] * 1.01)
    gap_down  = bool(today["Open"]   < prior["Close"] * 0.99)
    close_pct = (today["Close"] - today["Open"]) / today["Open"]
    near_high = bool(today["Close"] > past["High"].iloc[-20:].max() * 0.97)
    near_low  = bool(today["Close"] < past["Low"].iloc[-20:].min()  * 1.03)

    recent   = past.iloc[-10:]
    up_vol   = recent.loc[recent["Close"] > recent["Open"], "Volume"].sum()
    down_vol = recent.loc[recent["Close"] < recent["Open"], "Volume"].sum()
    total_v  = up_vol + down_vol
    vpin     = round(abs(up_vol - down_vol) / total_v, 3) if total_v > 0 else 0.5
    vpin_bull = vpin >= VPIN_THRESHOLD and up_vol   > down_vol
    vpin_bear = vpin >= VPIN_THRESHOLD and down_vol > up_vol

    lt, ls = [], []
    if close_pct > 0.01 and vol_spike and near_high:
        cp = min(close_pct * 100 * (today["Volume"] / avg_vol), 10.0)
        if cp >= CALL_PUT_RATIO_MIN:
            lt.append("cp_ratio_proxy"); ls.append(min(cp / (CALL_PUT_RATIO_MIN * 2), 1.0))
    if vol_spike:
        lt.append("vol_spike"); ls.append(min(today["Volume"] / (avg_vol * 3), 1.0))
    if block and not gap_down:
        lt.append("block_trade"); ls.append(0.5)
    if vpin_bull:
        lt.append("vpin_bullish"); ls.append(vpin)
    if gap_up:
        lt.append("gap_up"); ls.append(0.8)

    st, ss = [], []
    if close_pct < -0.01 and vol_spike and near_low:
        pp = min(abs(close_pct) * 100 * (today["Volume"] / avg_vol), 10.0)
        if pp >= CALL_PUT_RATIO_MIN:
            st.append("put_ratio_proxy"); ss.append(min(pp / (CALL_PUT_RATIO_MIN * 2), 1.0))
    if vol_spike and close_pct < 0:
        st.append("vol_spike_down"); ss.append(min(today["Volume"] / (avg_vol * 3), 1.0))
    if block and not gap_up and close_pct < -0.005:
        st.append("block_sell"); ss.append(0.5)
    if vpin_bear:
        st.append("vpin_bearish"); ss.append(vpin)
    if gap_down:
        st.append("gap_down"); ss.append(0.8)

    long_score  = round(sum(ls)/len(ls), 3) if ls else 0.0
    short_score = round(sum(ss)/len(ss), 3) if ss else 0.0

    return lt, long_score, st, short_score

# ── Path extraction ────────────────────────────────────────────────────────────

def extract_path(ticker: str, entry_date: datetime, entry_price: float,
                 direction: str, intraday: pd.DataFrame) -> np.ndarray:
    """
    Returns a float array of length BARS_AHEAD with % return from entry.
    NaN where data is missing. Shorts are inverted so +% = profitable.
    """
    bars = intraday[intraday.index >= pd.Timestamp(entry_date)].iloc[:BARS_AHEAD]
    if len(bars) < 5:
        return None

    path = (bars["Close"].values / entry_price - 1.0) * 100.0
    if direction == "short":
        path = -path

    out = np.full(BARS_AHEAD, np.nan)
    out[:len(path)] = path
    return out

# ── Main collection loop ───────────────────────────────────────────────────────

def collect_paths(daily: dict, intraday: dict, days_back: int) -> dict:
    """
    Returns dict: signal_name -> list of np.ndarray paths.
    Each trade's path appears in every group for each signal it triggered.
    """
    scan_end   = datetime.now() - timedelta(days=1)
    scan_start = scan_end - timedelta(days=days_back)

    sample       = next(iter(daily))
    trading_days = [
        d.to_pydatetime() for d in daily[sample].index
        if scan_start <= d.to_pydatetime() <= scan_end
    ]

    print(f"Scanning {len(trading_days)} trading days for signals...")
    groups = {}   # signal_name -> [path, ...]
    total  = 0

    for day in trading_days:
        for ticker in WATCHLIST:
            if ticker not in daily or ticker not in intraday:
                continue

            result = compute_signals(day, daily[ticker])
            if result is None:
                continue
            lt, lscore, st, sscore = result

            # Process long candidate
            if len(lt) >= LONG_MIN_SIGNALS and lscore >= LONG_SCORE_MIN:
                future = daily[ticker][daily[ticker].index > pd.Timestamp(day)]
                if not future.empty:
                    ed = future.index[0].to_pydatetime()
                    ep = float(future.iloc[0]["Open"])
                    path = extract_path(ticker, ed, ep, "long", intraday[ticker])
                    if path is not None:
                        for sig in lt:
                            groups.setdefault(sig, []).append(path)
                        # also store the full combo key
                        combo = "+".join(sorted(lt))
                        groups.setdefault(combo, []).append(path)
                        total += 1

            # Process short candidate
            if len(st) >= SHORT_MIN_SIGNALS and sscore >= SHORT_SCORE_MIN:
                future = daily[ticker][daily[ticker].index > pd.Timestamp(day)]
                if not future.empty:
                    ed = future.index[0].to_pydatetime()
                    ep = float(future.iloc[0]["Open"])
                    path = extract_path(ticker, ed, ep, "short", intraday[ticker])
                    if path is not None:
                        for sig in st:
                            groups.setdefault(sig, []).append(path)
                        combo = "+".join(sorted(st))
                        groups.setdefault(combo, []).append(path)
                        total += 1

    print(f"  {total} signal occurrences collected across {len(groups)} groups\n")
    return groups

# ── Plotting ───────────────────────────────────────────────────────────────────

SIGNAL_COLORS = {
    "gap_up":         "#3fb950",
    "gap_down":       "#f85149",
    "vpin_bullish":   "#58a6ff",
    "vpin_bearish":   "#d2a8ff",
    "vol_spike":      "#ffa657",
    "vol_spike_down": "#ff7b72",
    "cp_ratio_proxy": "#79c0ff",
    "put_ratio_proxy":"#f0883e",
    "block_trade":    "#56d364",
    "block_sell":     "#ffa198",
}

def plot_paths(groups: dict):
    # Filter to individual signals only (no combos) with enough trades
    individual = {k: v for k, v in groups.items()
                  if "+" not in k and len(v) >= MIN_TRADES}
    combos     = {k: v for k, v in groups.items()
                  if "+" in k and len(v) >= MIN_TRADES}

    all_groups = dict(sorted(individual.items(),
                             key=lambda x: len(x[1]), reverse=True))
    # Add top combos
    top_combos = dict(sorted(combos.items(),
                              key=lambda x: len(x[1]), reverse=True)[:4])
    all_groups.update(top_combos)

    n = len(all_groups)
    if n == 0:
        print("No groups with enough trades to plot."); return

    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 4),
                              facecolor="#0d1117")
    axes_flat = axes.flatten() if n > 1 else [axes]

    x = np.arange(BARS_AHEAD)
    # Day boundary bars
    day_lines = [BARS_PER_DAY * i for i in range(1, HOLD_DAYS)]

    for ax, (sig, paths) in zip(axes_flat, all_groups.items()):
        arr = np.array(paths)   # shape: (n_trades, BARS_AHEAD)
        avg = np.nanmean(arr, axis=0)
        p25 = np.nanpercentile(arr, 25, axis=0)
        p75 = np.nanpercentile(arr, 75, axis=0)

        col = SIGNAL_COLORS.get(sig.split("+")[0], "#58a6ff")

        ax.set_facecolor("#0d1117")
        ax.tick_params(colors="#cdd5df")
        for spine in ax.spines.values():
            spine.set_color("#30363d")

        ax.fill_between(x, p25, p75, alpha=0.2, color=col)
        ax.plot(x, avg, color=col, lw=2, label=f"avg (n={len(paths)})")
        ax.axhline(0, color="#444", lw=0.8, ls=":")

        for dl in day_lines:
            ax.axvline(dl, color="#444", lw=0.8, ls="--")

        # Mark peak bar
        valid = ~np.isnan(avg)
        if valid.any():
            peak_bar = int(np.argmax(avg[valid]))
            peak_val = avg[valid][peak_bar]
            ax.annotate(f"peak\nbar {peak_bar}",
                        xy=(peak_bar, peak_val),
                        xytext=(peak_bar + 8, peak_val + 0.05),
                        fontsize=7, color="#e6edf3",
                        arrowprops=dict(arrowstyle="->", color="#888", lw=0.8))

        ax.set_title(sig, color="#e6edf3", fontsize=10, pad=6)
        ax.set_xlabel(f"5-min bars from entry  "
                      f"(|day 1|{'|day 2|' if HOLD_DAYS>1 else ''}{'|day 3' if HOLD_DAYS>2 else ''})",
                      color="#8b949e", fontsize=7)
        ax.set_ylabel("% return (direction-normalised)", color="#8b949e", fontsize=7)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f%%"))
        ax.legend(facecolor="#0d1117", edgecolor="#30363d", labelcolor="#cdd5df", fontsize=8)

    # Hide unused subplots
    for ax in axes_flat[len(all_groups):]:
        ax.set_visible(False)

    plt.suptitle("Average Price Path by Signal Type  (entry = 0%, shorts inverted)",
                 color="#e6edf3", fontsize=13, y=1.01)
    plt.tight_layout(pad=2)

    out = DATA_DIR / "signal_path_analysis.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="#0d1117")
    plt.show()
    print(f"Chart saved to {out}")

    # ── Print peak summary table ───────────────────────────────────────────────
    print("\n  Signal peak summary")
    print(f"  {'Signal':<30} {'Trades':>7} {'Avg peak bar':>13} "
          f"{'Peak % return':>14} {'Avg final %':>12}")
    print("  " + "-" * 80)
    for sig, paths in sorted(all_groups.items(), key=lambda x: len(x[1]), reverse=True):
        arr  = np.array(paths)
        avg  = np.nanmean(arr, axis=0)
        valid = ~np.isnan(avg)
        if not valid.any():
            continue
        peak_bar = int(np.argmax(avg[valid]))
        peak_val = float(avg[valid][peak_bar])
        final    = float(avg[valid][-1]) if valid.any() else float("nan")
        print(f"  {sig:<30} {len(paths):>7} {peak_bar:>13} "
              f"{peak_val:>+13.3f}% {final:>+11.3f}%")

# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    no_cache  = "--no-cache" in sys.argv
    days_back = DAYS_BACK

    if "--days" in sys.argv:
        idx       = sys.argv.index("--days")
        days_back = int(sys.argv[idx + 1])

    daily    = load_daily(days_back, force=no_cache)
    intraday = load_intraday(days_back, force=no_cache)
    groups   = collect_paths(daily, intraday, days_back)
    plot_paths(groups)
