"""
backtest_7.py — Long/Short backtest with 5-year Alpaca intraday stops
======================================================================
Like backtest_6 but uses Alpaca Markets IEX feed for 5-minute bars,
allowing INTRADAY_STOPS over a full 5-year horizon.
OMXS30 removed (Alpaca is US-only).

Usage:
    python backtest_7.py                          # last 30 days
    python backtest_7.py --days 1825              # full 5 years
    python backtest_7.py --start 2021-01-01 --end 2026-01-01
    python backtest_7.py --long-only --chart
"""

import os
import sys
import json
import logging
import pandas as pd
import yfinance as yf
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from tabulate import tabulate
from colorama import Fore, Style, init

init(autoreset=True)

# ── Config ─────────────────────────────────────────────────────────────────────

# Alpaca API credentials — set ALPACA_API_KEY / ALPACA_API_SECRET as env vars (never commit keys)
ALPACA_API_KEY    = os.environ.get("ALPACA_API_KEY",    "")
ALPACA_API_SECRET = os.environ.get("ALPACA_API_SECRET", "")

# Signal detection
CALL_PUT_RATIO_MIN   = 3.0
VPIN_THRESHOLD       = 0.65

# ── Long parameters ──
LONG_SCORE_MIN       = 0.6
LONG_MIN_SIGNALS     = 2
LONG_HOLD_DAYS       = 3
LONG_STOP_LOSS_PCT   = 0.01
LONG_TAKE_PROFIT_PCT = 1.0

# ── Short parameters ──
SHORT_SCORE_MIN      = 0.6
SHORT_MIN_SIGNALS    = 2
SHORT_HOLD_DAYS      = 2
SHORT_SAME_DAY_EXIT  = True
SHORT_STOP_LOSS_PCT  = 0.01
SHORT_TAKE_PROFIT_PCT= 1.0

# ── Portfolio / capital limits ──
TOTAL_CAPITAL        = 25_000
MAX_POSITIONS        = 2
COMPOUNDING          = True
POSITION_SIZE_USD    = TOTAL_CAPITAL / MAX_POSITIONS

MACRO_EVENT_THRESHOLD = 8

# True = require at least one directional signal (gap/VPIN/ratio)
REQUIRE_DIRECTIONAL = True

# True = use 5-minute Alpaca bars for SL/TP simulation, skipping the
# first bar of the entry day (09:30-09:35 ET) to avoid opening noise.
INTRADAY_STOPS = True

WATCHLIST_DEFAULT = [
    "NVDA","MSFT","AAPL","AMZN","META","GOOGL","TSLA","JPM",
    "XOM","PFE","MRNA","AMD","NFLX","CRM","INTC","BAC","GS",
    "ABBV","LLY","UNH","V","MA","AVGO","ORCL","ADBE",
]

WATCHLIST_SP50 = [
    "NVDA","AAPL","MSFT","AMZN","GOOGL","META","TSLA","AVGO",
    "BRK-B","JPM","LLY","V","UNH","XOM","WMT","ORCL","MA",
    "COST","NFLX","JNJ","HD","BAC","PG","ABBV","CRM","MRK",
    "KO","AMD","ADBE","GS","NOW","T","MCD","PEP","DIS","IBM",
    "CSCO","PM","TMO","RTX","AXP","INTU","CAT","VZ","ISRG",
    "AMGN","QCOM","SPGI","TXN","PFE",
]

WATCHLIST = WATCHLIST_DEFAULT

DATA_DIR        = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)
BT_TRADES_FILE  = DATA_DIR / "backtest_trades_7.json"
BT_SUMMARY_FILE = DATA_DIR / "backtest_summary_7.csv"

import io
if hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'buffer'):
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[
        logging.FileHandler(DATA_DIR / "backtest_7.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

# ── Data classes ───────────────────────────────────────────────────────────────

@dataclass
class BTSignal:
    ticker: str
    date: str
    direction: str
    composite_score: float
    signals_triggered: list
    is_macro_event: bool

@dataclass
class BTTrade:
    ticker: str
    direction: str
    entry_date: str
    entry_price: float
    exit_date: str
    exit_price: float
    hold_days: int
    pnl_pct: float
    pnl_usd: float
    exit_reason: str
    signals: list
    composite_score: float
    is_macro_event: bool

# ── History fetch ──────────────────────────────────────────────────────────────

def fetch_all_history(days_back: int) -> dict:
    end   = datetime.now()
    start = end - timedelta(days=days_back + 60)
    log.info(f"Downloading daily price history for {len(WATCHLIST)} tickers...")
    raw = yf.download(
        WATCHLIST,
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        interval="1d",
        group_by="ticker",
        auto_adjust=True,
        progress=False,
    )
    history = {}
    for ticker in WATCHLIST:
        try:
            df = raw[ticker].dropna(how="all").copy() if len(WATCHLIST) > 1 \
                 else raw.dropna(how="all").copy()
            df.index = pd.to_datetime(df.index).tz_localize(None)
            history[ticker] = df
        except Exception as e:
            log.warning(f"  {ticker}: failed - {e}")
    log.info(f"  Loaded {len(history)} tickers\n")
    return history


def fetch_intraday_alpaca(start: datetime, end: datetime) -> dict:
    """Fetch 5-min bars from Alpaca IEX feed for the full backtest window."""
    try:
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
    except ImportError:
        log.error("alpaca-py not installed. Run: pip install alpaca-py")
        return {}

    log.info(f"Downloading 5-min Alpaca bars ({start.date()} to {end.date()}) "
             f"for {len(WATCHLIST)} tickers...")

    client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_API_SECRET)

    request = StockBarsRequest(
        symbol_or_symbols=WATCHLIST,
        timeframe=TimeFrame(5, TimeFrameUnit.Minute),
        start=start,
        end=end,
        feed="iex",
        limit=None,
    )

    try:
        bars = client.get_stock_bars(request)
        raw  = bars.df
    except Exception as e:
        log.error(f"Alpaca fetch failed: {e}")
        return {}

    history = {}
    for ticker in WATCHLIST:
        try:
            if ticker in raw.index.get_level_values("symbol"):
                df = raw.xs(ticker, level="symbol").copy()
                df.index = pd.to_datetime(df.index).tz_localize(None)
                df.rename(columns={
                    "open": "Open", "high": "High",
                    "low": "Low", "close": "Close", "volume": "Volume"
                }, inplace=True)
                history[ticker] = df
        except Exception as e:
            log.warning(f"  {ticker}: intraday parse failed - {e}")

    log.info(f"  Loaded {len(history)} tickers (intraday)\n")
    return history

# ── Signal computation ─────────────────────────────────────────────────────────

def compute_signals(ticker: str, date: datetime, history: pd.DataFrame):
    past = history[history.index <= pd.Timestamp(date)].copy()
    if len(past) < 25:
        return None, None

    today   = past.iloc[-1]
    prior   = past.iloc[-2]
    avg_vol = past["Volume"].iloc[-21:-1].mean()

    vol_spike   = bool(today["Volume"] > avg_vol * 2.0)
    block       = bool(today["Volume"] > avg_vol * 1.5)
    gap_up      = bool(today["Open"]   > prior["Close"] * 1.01)
    gap_down    = bool(today["Open"]   < prior["Close"] * 0.99)
    close_pct   = (today["Close"] - today["Open"]) / today["Open"]
    recent_high = past["High"].iloc[-20:].max()
    recent_low  = past["Low"].iloc[-20:].min()
    near_high   = bool(today["Close"] > recent_high * 0.97)
    near_low    = bool(today["Close"] < recent_low  * 1.03)

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

    long_base   = round(sum(ls) / len(ls), 3) if ls else 0.0
    short_base  = round(sum(ss) / len(ss), 3) if ss else 0.0
    long_bonus  = min(max(len(lt) - LONG_MIN_SIGNALS,  0) * 0.04, 0.12)
    short_bonus = min(max(len(st) - SHORT_MIN_SIGNALS, 0) * 0.04, 0.12)
    long_score  = round(min(long_base  + long_bonus,  1.0), 3)
    short_score = round(min(short_base + short_bonus, 1.0), 3)

    _long_dir  = {"cp_ratio_proxy", "vpin_bullish", "gap_up"}
    _short_dir = {"put_ratio_proxy", "vpin_bearish", "gap_down"}
    long_directional  = not REQUIRE_DIRECTIONAL or bool(set(lt) & _long_dir)
    short_directional = not REQUIRE_DIRECTIONAL or bool(set(st) & _short_dir)

    long_sig  = BTSignal(ticker, date.strftime("%Y-%m-%d"), "long",  long_score,  lt, False) \
                if long_score  >= LONG_SCORE_MIN  and len(lt) >= LONG_MIN_SIGNALS  and long_directional  else None
    short_sig = BTSignal(ticker, date.strftime("%Y-%m-%d"), "short", short_score, st, False) \
                if short_score >= SHORT_SCORE_MIN and len(st) >= SHORT_MIN_SIGNALS and short_directional else None

    return long_sig, short_sig

# ── Trade simulation ───────────────────────────────────────────────────────────

def simulate_trade(ticker: str, direction: str, entry_date: datetime,
                   entry_price: float, history: pd.DataFrame,
                   signal: BTSignal, pos_size: float = POSITION_SIZE_USD,
                   intraday: pd.DataFrame = None) -> BTTrade:
    exit_price  = entry_price
    exit_date   = entry_date
    exit_reason = "closed_time"
    days_held   = 0

    if direction == "long":
        hold     = LONG_HOLD_DAYS
        stop_p   = entry_price * (1 - LONG_STOP_LOSS_PCT)
        target_p = entry_price * (1 + LONG_TAKE_PROFIT_PCT)
    else:
        hold     = 1 if SHORT_SAME_DAY_EXIT else SHORT_HOLD_DAYS
        stop_p   = entry_price * (1 + SHORT_STOP_LOSS_PCT)
        target_p = entry_price * (1 - SHORT_TAKE_PROFIT_PCT)

    if intraday is not None and not intraday.empty:
        daily_from_entry = history[history.index >= pd.Timestamp(entry_date)]
        if not daily_from_entry.empty:
            hold_end   = daily_from_entry.index[min(hold - 1, len(daily_from_entry) - 1)]
            future_5m  = intraday[
                (intraday.index >= pd.Timestamp(entry_date)) &
                (intraday.index.normalize() <= hold_end)
            ].copy()

            entry_day = pd.Timestamp(entry_date).normalize()
            for idx, row in future_5m.iterrows():
                exit_date  = idx.to_pydatetime()
                exit_price = float(row["Close"])
                if idx.normalize() == entry_day:
                    # Entry day: track price but don't trigger stops
                    continue

                if direction == "long":
                    if   row["Low"]  <= stop_p:   exit_price, exit_reason = stop_p,   "closed_sl"; break
                    elif row["High"] >= target_p: exit_price, exit_reason = target_p, "closed_tp"; break
                else:
                    if   row["High"] >= stop_p:   exit_price, exit_reason = stop_p,   "closed_sl"; break
                    elif row["Low"]  <= target_p: exit_price, exit_reason = target_p, "closed_tp"; break

            days_held = hold

    else:
        future = history[history.index > pd.Timestamp(entry_date)].copy()
        for i, (idx, row) in enumerate(future.iterrows()):
            if i >= hold:
                break
            days_held += 1
            exit_date  = idx.to_pydatetime()

            if direction == "long":
                if   row["Low"]  <= stop_p:   exit_price, exit_reason = stop_p,   "closed_sl"; break
                elif row["High"] >= target_p: exit_price, exit_reason = target_p, "closed_tp"; break
                else: exit_price = float(row["Close"])
            else:
                if   row["High"] >= stop_p:   exit_price, exit_reason = stop_p,   "closed_sl"; break
                elif row["Low"]  <= target_p: exit_price, exit_reason = target_p, "closed_tp"; break
                else: exit_price = float(row["Close"])

    if direction == "long":
        pnl_pct = round((exit_price - entry_price) / entry_price * 100, 3)
    else:
        pnl_pct = round((entry_price - exit_price) / entry_price * 100, 3)

    pnl_usd       = round(pos_size * pnl_pct / 100, 2)
    exit_date_str = exit_date.strftime("%Y-%m-%d") if isinstance(exit_date, datetime) \
                    else str(exit_date)[:10]

    return BTTrade(
        ticker=ticker, direction=direction,
        entry_date=entry_date.strftime("%Y-%m-%d"), entry_price=round(entry_price, 2),
        exit_date=exit_date_str, exit_price=round(exit_price, 2),
        hold_days=days_held, pnl_pct=pnl_pct, pnl_usd=pnl_usd,
        exit_reason=exit_reason, signals=signal.signals_triggered,
        composite_score=signal.composite_score, is_macro_event=signal.is_macro_event,
    )

# ── Main backtest loop ─────────────────────────────────────────────────────────

def run_backtest(days_back: int = 30, long_only: bool = False, short_only: bool = False,
                 start: str = None, end: str = None):
    sides = []
    if not short_only: sides.append("long")
    if not long_only:  sides.append("short")

    if start and end:
        scan_start = datetime.strptime(start, "%Y-%m-%d")
        scan_end   = datetime.strptime(end,   "%Y-%m-%d")
        fetch_days = (datetime.now() - scan_start).days + 60
    else:
        scan_end   = datetime.now() - timedelta(days=1)
        scan_start = scan_end - timedelta(days=days_back * 1.5)
        fetch_days = days_back

    log.info("=" * 65)
    log.info(f"BACKTEST_7  -  "
             f"{scan_start.strftime('%Y-%m-%d')} to {scan_end.strftime('%Y-%m-%d')}")
    log.info(f"Sides: {', '.join(sides)}")
    log.info(f"Intraday stops: {'Alpaca IEX 5-min' if INTRADAY_STOPS else 'OFF (daily bars)'}")
    log.info(f"Capital: ${TOTAL_CAPITAL:,} | Max positions: {MAX_POSITIONS}")
    log.info("=" * 65)

    all_history   = fetch_all_history(fetch_days)
    intraday_hist = fetch_intraday_alpaca(scan_start - timedelta(days=1), datetime.now()) \
                    if INTRADAY_STOPS else {}

    sample = next(iter(all_history))
    trading_days = [
        d.to_pydatetime() for d in all_history[sample].index
        if scan_start <= d.to_pydatetime() <= scan_end
    ]
    if not start:
        trading_days = trading_days[-days_back:]

    log.info(f"Scanning {len(trading_days)} days: "
             f"{trading_days[0].strftime('%Y-%m-%d')} to "
             f"{trading_days[-1].strftime('%Y-%m-%d')}\n")

    all_trades = []
    open_long  = {}
    open_short = {}
    portfolio  = []
    equity     = TOTAL_CAPITAL

    for day in trading_days:
        day_str = day.strftime("%Y-%m-%d")

        if COMPOUNDING:
            for t in [t for t in all_trades if t.get("exit_date", "") == day_str]:
                equity += t["pnl_usd"]
            equity = max(equity, 1)

        pos_size   = (equity / MAX_POSITIONS) if COMPOUNDING else POSITION_SIZE_USD
        portfolio  = [p for p in portfolio if p["exit_date"] > day + timedelta(days=1)]
        slots_used = len(portfolio)
        slots_free = MAX_POSITIONS - slots_used

        day_longs  = []
        day_shorts = []

        for ticker in WATCHLIST:
            if ticker not in all_history:
                continue
            long_sig, short_sig = compute_signals(ticker, day, all_history[ticker])
            if long_sig  and "long"  in sides:
                if not (ticker in open_long and
                        (day - open_long[ticker]).days < LONG_HOLD_DAYS):
                    day_longs.append((ticker, long_sig))
            if short_sig and "short" in sides:
                if not (ticker in open_short and
                        (day - open_short[ticker]).days < SHORT_HOLD_DAYS):
                    day_shorts.append((ticker, short_sig))

        is_macro_long  = len(day_longs)  >= MACRO_EVENT_THRESHOLD
        is_macro_short = len(day_shorts) >= MACRO_EVENT_THRESHOLD
        for _, s in day_longs:  s.is_macro_event = is_macro_long
        for _, s in day_shorts: s.is_macro_event = is_macro_short

        all_candidates = [(t, s, "long")  for t, s in day_longs] + \
                         [(t, s, "short") for t, s in day_shorts]
        all_candidates.sort(key=lambda x: x[1].composite_score, reverse=True)

        day_trades  = []
        slots_taken = 0

        for ticker, signal, direction in all_candidates:
            if (slots_used + slots_taken) >= MAX_POSITIONS or slots_taken >= slots_free:
                skipped = len(all_candidates) - all_candidates.index((ticker, signal, direction))
                log.info(f"  [portfolio full] {skipped} signal(s) skipped")
                break

            future = all_history[ticker][all_history[ticker].index > pd.Timestamp(day)]
            if future.empty:
                continue

            ep = float(future.iloc[0]["Open"])
            ed = future.index[0].to_pydatetime()

            intraday = intraday_hist.get(ticker) if INTRADAY_STOPS else None
            trade = simulate_trade(ticker, direction, ed, ep, all_history[ticker], signal, pos_size, intraday)
            all_trades.append(asdict(trade))
            slots_taken += 1

            actual_exit = datetime.strptime(trade.exit_date, "%Y-%m-%d") + timedelta(days=1)
            portfolio.append({"exit_date": actual_exit, "ticker": ticker, "direction": direction})

            if direction == "long":  open_long[ticker]  = day
            else:                    open_short[ticker] = day

            day_trades.append((trade, signal))

        if day_trades:
            macro_tag = ""
            if is_macro_long  and day_longs:  macro_tag = " [MACRO - broad rally]"
            if is_macro_short and day_shorts: macro_tag = " [MACRO - broad sell-off]"
            log.info(f"{day_str}:{macro_tag}")
            for trade, sig in sorted(day_trades, key=lambda x: x[0].direction):
                col  = Fore.GREEN if trade.pnl_usd >= 0 else Fore.RED
                dcol = Fore.CYAN if trade.direction == "long" else Fore.MAGENTA
                dlbl = "L" if trade.direction == "long" else "S"
                print(f"  {dcol}[{dlbl}]{Style.RESET_ALL} "
                      f"{col}{trade.ticker:5s}{Style.RESET_ALL} "
                      f"score={sig.composite_score:.2f} "
                      f"sigs={sig.signals_triggered} "
                      f"-> {trade.exit_reason.replace('closed_','')} "
                      f"{trade.pnl_pct:+.1f}% (${trade.pnl_usd:+.2f})")
        else:
            log.info(f"{day_str}: no signals")

    with open(BT_TRADES_FILE, "w", encoding="utf-8") as f:
        json.dump(all_trades, f, indent=2, default=str)

    label = f"{scan_start.strftime('%Y-%m-%d')} to {scan_end.strftime('%Y-%m-%d')}"
    print_summary(all_trades, label, sides)
    return all_trades

# ── Summary ────────────────────────────────────────────────────────────────────

def print_summary(trades: list, label: str, sides: list):
    if not trades:
        print(f"\n{Fore.YELLOW}No trades triggered.")
        return

    df = pd.DataFrame(trades)

    def section(subset, direction_label, colour):
        if subset.empty: return
        wins   = subset[subset["pnl_usd"] > 0]
        losses = subset[subset["pnl_usd"] <= 0]
        wr     = len(wins) / len(subset) * 100
        avg_w  = wins["pnl_usd"].mean()   if len(wins)   > 0 else 0
        avg_l  = losses["pnl_usd"].mean() if len(losses) > 0 else 0
        pf     = abs(avg_w / avg_l)       if avg_l != 0      else 0
        sharpe = (subset["pnl_pct"].mean() / subset["pnl_pct"].std()) \
                 if subset["pnl_pct"].std() > 0 else 0
        total      = subset["pnl_usd"].sum()
        macro_pnl  = subset[subset["is_macro_event"]]["pnl_usd"].sum()
        specific   = subset[~subset["is_macro_event"]]["pnl_usd"].sum()

        print(f"\n{colour}{'='*65}")
        print(f"  {label}  |  {direction_label} RESULTS")
        print(f"{'='*65}{Style.RESET_ALL}")
        stats = [
            ["Total trades",         len(subset)],
            ["Win rate",             f"{wr:.1f}%  ({len(wins)}W / {len(losses)}L)"],
            ["Avg return/trade",     f"{subset['pnl_pct'].mean():+.2f}%"],
            ["Avg winning trade",    f"${avg_w:+.2f}"],
            ["Avg losing trade",     f"${avg_l:+.2f}"],
            ["Profit factor",        f"{pf:.2f}"],
            ["Sharpe ratio",         f"{sharpe:.2f}"],
            ["Total P&L",            f"${total:+.2f}"],
            ["  macro event P&L",    f"${macro_pnl:+.2f}"],
            ["  stock-specific P&L", f"${specific:+.2f}"],
            ["Stopped out",          int((subset["exit_reason"]=="closed_sl").sum())],
            ["Hit take profit",      int((subset["exit_reason"]=="closed_tp").sum())],
            ["Exited by time",       int((subset["exit_reason"]=="closed_time").sum())],
        ]
        print(tabulate(stats, tablefmt="simple", colalign=("left","right")))

        tkr = subset.groupby("ticker").agg(
            trades=("pnl_usd","count"),
            total=("pnl_usd","sum"),
            avg_pct=("pnl_pct","mean"),
            wins=("pnl_usd", lambda x: (x>0).sum()),
        ).reset_index()
        tkr["wr"] = (tkr["wins"] / tkr["trades"] * 100).round(0)
        tkr = tkr.sort_values("total", ascending=False)
        print(f"\n  Per-Ticker:")
        rows = [[r["ticker"], int(r["trades"]), f"${r['total']:+.2f}",
                 f"{r['avg_pct']:+.2f}%", f"{r['wr']:.0f}%"]
                for _, r in tkr.iterrows()]
        print(tabulate(rows, headers=["Ticker","Trades","P&L","Avg%","Win%"], tablefmt="simple"))

    longs  = df[df["direction"]=="long"]
    shorts = df[df["direction"]=="short"]

    section(longs,  "LONG",  Fore.CYAN)
    section(shorts, "SHORT", Fore.MAGENTA)

    print(f"\n{Fore.WHITE}{'='*65}")
    print(f"  {label}  |  COMBINED LONG + SHORT")
    print(f"{'='*65}{Style.RESET_ALL}")

    total_wr  = len(df[df["pnl_usd"]>0]) / len(df) * 100
    long_pnl  = longs["pnl_usd"].sum()  if not longs.empty  else 0
    short_pnl = shorts["pnl_usd"].sum() if not shorts.empty else 0
    macro_pnl = df[df["is_macro_event"]]["pnl_usd"].sum()
    specific  = df[~df["is_macro_event"]]["pnl_usd"].sum()
    total_return_pct = (long_pnl + short_pnl) / TOTAL_CAPITAL * 100

    print(tabulate([
        ["Total capital",             f"${TOTAL_CAPITAL:,}"],
        ["Mode",                      "Compounding" if COMPOUNDING else "Fixed sizing"],
        ["Max positions (slots)",     MAX_POSITIONS],
        ["Intraday stops",            "Alpaca IEX 5-min" if INTRADAY_STOPS else "OFF"],
        ["Total trades",              len(df)],
        ["Long trades",               len(longs)],
        ["Short trades",              len(shorts)],
        ["Combined win rate",         f"{total_wr:.1f}%"],
        ["Long P&L",                  f"${long_pnl:+.2f}"],
        ["Short P&L",                 f"${short_pnl:+.2f}"],
        ["Total P&L",                 f"${long_pnl+short_pnl:+.2f}"],
        ["Return on capital",         f"{total_return_pct:+.2f}%"],
        ["Macro event P&L",           f"${macro_pnl:+.2f}"],
        ["Stock-specific signal P&L", f"${specific:+.2f}"],
    ], tablefmt="simple", colalign=("left","right")))

    df.to_csv(BT_SUMMARY_FILE, index=False, encoding="utf-8")
    print(f"\n  Saved to {BT_SUMMARY_FILE}\n")

# ── Chart ──────────────────────────────────────────────────────────────────────

def plot_results():
    try:
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
    except ImportError:
        print("pip install matplotlib"); return

    if not BT_TRADES_FILE.exists():
        print("Run backtest first."); return
    with open(BT_TRADES_FILE, encoding="utf-8") as f:
        trades = json.load(f)
    if not trades:
        print("No trades."); return

    df = pd.DataFrame(trades).sort_values("entry_date")
    df["entry_date"]    = pd.to_datetime(df["entry_date"])
    df["cumulative_pnl"] = df["pnl_usd"].cumsum()

    longs  = df[df["direction"]=="long"].copy()
    shorts = df[df["direction"]=="short"].copy()
    if not longs.empty:  longs["cum"]  = longs["pnl_usd"].cumsum()
    if not shorts.empty: shorts["cum"] = shorts["pnl_usd"].cumsum()

    fig, axes = plt.subplots(3, 1, figsize=(13, 11), facecolor="#0d0d0d")
    for ax in axes:
        ax.set_facecolor("#111"); ax.tick_params(colors="#888"); ax.spines[:].set_color("#333")

    ax = axes[0]
    ax.plot(df["entry_date"], df["cumulative_pnl"], color="#378ADD", lw=2, label="Combined")
    if not longs.empty:  ax.plot(longs["entry_date"],  longs["cum"],  color="#1D9E75", lw=1, ls="--", label="Longs")
    if not shorts.empty: ax.plot(shorts["entry_date"], shorts["cum"], color="#D85A30", lw=1, ls="--", label="Shorts")
    ax.axhline(0, color="#555", lw=0.8, ls=":")
    ax.set_title("Cumulative P&L — Backtest 7 (Alpaca 5-min stops)", color="#ccc", pad=8)
    ax.set_ylabel("USD", color="#888")
    ax.legend(facecolor="#111", edgecolor="#333", labelcolor="#ccc", fontsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))

    for ax, subset, title in [
        (axes[1], longs,  "Long Trades"),
        (axes[2], shorts, "Short Trades"),
    ]:
        if not subset.empty:
            colors = ["#1D9E75" if p >= 0 else "#D85A30" for p in subset["pnl_usd"]]
            ax.bar(range(len(subset)), subset["pnl_usd"], color=colors)
            ax.set_xticks(range(len(subset)))
            ax.set_xticklabels(subset["ticker"].tolist(), rotation=45, ha="right", fontsize=7, color="#888")
        ax.axhline(0, color="#555", lw=0.8)
        ax.set_title(f"{title} - Per-Trade P&L", color="#ccc", pad=8)
        ax.set_ylabel("USD", color="#888")

    plt.tight_layout(pad=2)
    out = DATA_DIR / "backtest_7_chart.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="#0d0d0d")
    plt.show()
    print(f"Chart saved to {out}")

# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    days       = 30
    long_only  = "--long-only"  in sys.argv
    short_only = "--short-only" in sys.argv
    chart      = "--chart"      in sys.argv
    start_date = None
    end_date   = None

    if "--days" in sys.argv:
        idx  = sys.argv.index("--days")
        days = int(sys.argv[idx + 1])

    if "--start" in sys.argv:
        idx        = sys.argv.index("--start")
        start_date = sys.argv[idx + 1]

    if "--end" in sys.argv:
        idx      = sys.argv.index("--end")
        end_date = sys.argv[idx + 1]

    trades = run_backtest(
        days_back=days,
        long_only=long_only,
        short_only=short_only,
        start=start_date,
        end=end_date,
    )
    if chart and trades:
        plot_results()
