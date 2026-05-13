"""
backtest.py — Long/Short historical backtest
=============================================
Replays the last 30 trading days using both long (gap-up, bullish flow)
and short (gap-down, bearish flow) signals, making the strategy
market-neutral and less dependent on overall market direction.

Usage:
    python backtest.py                  # last 30 days, both sides
    python backtest.py --days 60        # last 60 days
    python backtest.py --long-only      # long trades only
    python backtest.py --short-only     # short trades only
    python backtest.py --chart          # show matplotlib chart after

Short selling mechanics:
    - Entry: sell short at next day open after signal fires
    - Profit when price falls (P&L = entry - exit)
    - Stop loss: price RISES 5% above entry (capped loss)
    - Take profit: price FALLS 10% below entry
    - Same 3-day hold period as longs
"""

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

# ── Config ─────────────────────────────────────────────────────────────────────────────────

# Signal detection
CALL_PUT_RATIO_MIN   = 3.0
VPIN_THRESHOLD       = 0.65

# ── Long parameters ──
LONG_SCORE_MIN       = 0.5   # minimum composite score to open a long
LONG_MIN_SIGNALS     = 2      # minimum signals triggered
LONG_HOLD_DAYS       = 3      # days to hold before time-exit
LONG_STOP_LOSS_PCT   = 0.01   # stop out if price drops this much
LONG_TAKE_PROFIT_PCT = 0.10   # take profit if price rises this much

# ── Short parameters (tuned independently) ──
SHORT_SCORE_MIN      = 0.5   # higher bar - shorts need stronger confirmation
SHORT_MIN_SIGNALS    = 2      # require at least 2 signals for shorts
SHORT_HOLD_DAYS      = 2      # shorter hold - shorts can reverse fast
SHORT_STOP_LOSS_PCT  = 0.01   # stop out if price rises this much
SHORT_TAKE_PROFIT_PCT= 0.10   # take profit if price falls this much

# ── Portfolio / capital limits ──
TOTAL_CAPITAL        = 25_000  # total account size in USD
MAX_POSITIONS        = 5      # maximum simultaneous open slots
COMPOUNDING         = True    # True = position size grows/shrinks with equity
POSITION_SIZE_USD    = TOTAL_CAPITAL / MAX_POSITIONS  # used only when COMPOUNDING=False

# If more than this many tickers signal same direction on one day = macro event
MACRO_EVENT_THRESHOLD = 8

WATCHLIST = [
    "NVDA","MSFT","AAPL","AMZN","META","GOOGL","TSLA","JPM",
    "XOM","PFE","MRNA","AMD","NFLX","CRM","INTC","BAC","GS",
    "ABBV","LLY","UNH","V","MA","AVGO","ORCL","ADBE",
]

DATA_DIR        = Path("./data")
DATA_DIR.mkdir(exist_ok=True)
BT_TRADES_FILE  = DATA_DIR / "backtest_trades_ls.json"
BT_SUMMARY_FILE = DATA_DIR / "backtest_summary_ls.csv"

# Fix Windows console encoding
import io
if hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'buffer'):
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[
        logging.FileHandler(DATA_DIR / "backtest_ls.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

# ── Data classes ───────────────────────────────────────────────────────────────

@dataclass
class BTSignal:
    ticker: str
    date: str
    direction: str          # "long" or "short"
    composite_score: float
    signals_triggered: list
    is_macro_event: bool    # True if many tickers fired same day

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
    log.info(f"Downloading price history for {len(WATCHLIST)} tickers...")
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

# ── Signal computation ─────────────────────────────────────────────────────────

def compute_signals(ticker: str, date: datetime, history: pd.DataFrame):
    """
    Returns (long_signal, short_signal) — either can be None.
    No lookahead: only uses data up to and including date.
    """
    past = history[history.index <= pd.Timestamp(date)].copy()
    if len(past) < 25:
        return None, None

    today   = past.iloc[-1]
    prior   = past.iloc[-2]
    avg_vol = past["Volume"].iloc[-21:-1].mean()

    # Shared indicators
    vol_spike   = bool(today["Volume"] > avg_vol * 2.0)
    block       = bool(today["Volume"] > avg_vol * 1.5)
    gap_up      = bool(today["Open"]   > prior["Close"] * 1.01)
    gap_down    = bool(today["Open"]   < prior["Close"] * 0.99)
    close_pct   = (today["Close"] - today["Open"]) / today["Open"]
    recent_high = past["High"].iloc[-20:].max()
    recent_low  = past["Low"].iloc[-20:].min()
    near_high   = bool(today["Close"] > recent_high * 0.97)
    near_low    = bool(today["Close"] < recent_low  * 1.03)

    # 10-day directional VPIN
    recent   = past.iloc[-10:]
    up_vol   = recent.loc[recent["Close"] > recent["Open"], "Volume"].sum()
    down_vol = recent.loc[recent["Close"] < recent["Open"], "Volume"].sum()
    total_v  = up_vol + down_vol
    vpin     = round(abs(up_vol - down_vol) / total_v, 3) if total_v > 0 else 0.5
    vpin_bull = vpin >= VPIN_THRESHOLD and up_vol   > down_vol
    vpin_bear = vpin >= VPIN_THRESHOLD and down_vol > up_vol

    # ── LONG signals ──
    lt, ls = [], []

    if close_pct > 0.01 and vol_spike and near_high:
        cp = min(close_pct * 100 * (today["Volume"] / avg_vol), 10.0)
        if cp >= CALL_PUT_RATIO_MIN:
            lt.append("cp_ratio_proxy"); ls.append(min(cp / (CALL_PUT_RATIO_MIN * 2), 1.0))

    if vol_spike:
        lt.append("vol_spike"); ls.append(min(today["Volume"] / (avg_vol * 3), 1.0))

    if block and not gap_down:
        lt.append("block_trade"); ls.append(0.7)

    if vpin_bull:
        lt.append("vpin_bullish"); ls.append(vpin)

    if gap_up:
        lt.append("gap_up"); ls.append(0.6)

    # ── SHORT signals (mirror) ──
    st, ss = [], []

    if close_pct < -0.01 and vol_spike and near_low:
        pp = min(abs(close_pct) * 100 * (today["Volume"] / avg_vol), 10.0)
        if pp >= CALL_PUT_RATIO_MIN:
            st.append("put_ratio_proxy"); ss.append(min(pp / (CALL_PUT_RATIO_MIN * 2), 1.0))

    if vol_spike and close_pct < 0:
        st.append("vol_spike_down"); ss.append(min(today["Volume"] / (avg_vol * 3), 1.0))

    if block and not gap_up and close_pct < -0.005:
        st.append("block_sell"); ss.append(0.7)

    if vpin_bear:
        st.append("vpin_bearish"); ss.append(vpin)

    if gap_down:
        st.append("gap_down"); ss.append(0.6)

    long_score  = round(sum(ls) / len(ls), 3) if ls else 0.0
    short_score = round(sum(ss) / len(ss), 3) if ss else 0.0

    long_sig  = BTSignal(ticker, date.strftime("%Y-%m-%d"), "long",  long_score,  lt, False) \
                if long_score  >= LONG_SCORE_MIN  and len(lt) >= LONG_MIN_SIGNALS  else None
    short_sig = BTSignal(ticker, date.strftime("%Y-%m-%d"), "short", short_score, st, False) \
                if short_score >= SHORT_SCORE_MIN and len(st) >= SHORT_MIN_SIGNALS else None

    return long_sig, short_sig

# ── Trade simulation ───────────────────────────────────────────────────────────

def simulate_trade(ticker: str, direction: str, entry_date: datetime,
                   entry_price: float, history: pd.DataFrame,
                   signal: BTSignal, pos_size: float = POSITION_SIZE_USD) -> BTTrade:
    future      = history[history.index > pd.Timestamp(entry_date)].copy()
    exit_price  = entry_price
    exit_date   = entry_date
    exit_reason = "closed_time"
    days_held   = 0

    if direction == "long":
        hold     = LONG_HOLD_DAYS
        stop_p   = entry_price * (1 - LONG_STOP_LOSS_PCT)
        target_p = entry_price * (1 + LONG_TAKE_PROFIT_PCT)
    else:
        hold     = SHORT_HOLD_DAYS
        stop_p   = entry_price * (1 + SHORT_STOP_LOSS_PCT)   # loss if price rises
        target_p = entry_price * (1 - SHORT_TAKE_PROFIT_PCT) # profit if price falls

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
    """
    start / end: optional date strings "YYYY-MM-DD" to pin an exact window.
    If provided, days_back is ignored for the scan window but still used
    to fetch enough history for rolling-window indicators.
    """
    sides = []
    if not short_only: sides.append("long")
    if not long_only:  sides.append("short")

    # Resolve scan window
    if start and end:
        scan_start = datetime.strptime(start, "%Y-%m-%d")
        scan_end   = datetime.strptime(end,   "%Y-%m-%d")
        # fetch enough history before scan_start for rolling indicators (60 days buffer)
        fetch_days = (datetime.now() - scan_start).days + 60
    else:
        scan_end   = datetime.now() - timedelta(days=1)
        scan_start = scan_end - timedelta(days=days_back * 1.5)
        fetch_days = days_back

    log.info("=" * 65)
    log.info(f"LONG/SHORT BACKTEST  -  "
             f"{scan_start.strftime('%Y-%m-%d')} to {scan_end.strftime('%Y-%m-%d')}")
    log.info(f"Sides: {', '.join(sides)}")
    log.info(f"Capital: ${TOTAL_CAPITAL:,} | Max positions: {MAX_POSITIONS} | "
             f"Per slot: ${POSITION_SIZE_USD:.0f}")
    log.info(f"Long:  score>={LONG_SCORE_MIN}, signals>={LONG_MIN_SIGNALS}, "
             f"hold={LONG_HOLD_DAYS}d, SL={LONG_STOP_LOSS_PCT*100:.0f}%, "
             f"TP={LONG_TAKE_PROFIT_PCT*100:.0f}%")
    log.info(f"Short: score>={SHORT_SCORE_MIN}, signals>={SHORT_MIN_SIGNALS}, "
             f"hold={SHORT_HOLD_DAYS}d, SL={SHORT_STOP_LOSS_PCT*100:.0f}%, "
             f"TP={SHORT_TAKE_PROFIT_PCT*100:.0f}%")
    log.info("=" * 65)

    all_history  = fetch_all_history(fetch_days)
    sample       = next(iter(all_history))
    trading_days = [
        d.to_pydatetime() for d in all_history[sample].index
        if scan_start <= d.to_pydatetime() <= scan_end
    ]
    if not start:
        trading_days = trading_days[-days_back:]

    log.info(f"Scanning {len(trading_days)} days: "
             f"{trading_days[0].strftime('%Y-%m-%d')} to "
             f"{trading_days[-1].strftime('%Y-%m-%d')}\n")

    all_trades  = []
    open_long   = {}   # ticker -> entry_day (long positions)
    open_short  = {}   # ticker -> entry_day (short positions)
    portfolio   = []   # list of {"exit_date": datetime, "ticker": str, "direction": str}
    equity      = TOTAL_CAPITAL   # running equity (only changes when COMPOUNDING=True)

    for day in trading_days:
        day_str = day.strftime("%Y-%m-%d")

        # ── Update equity from trades that closed today ──
        if COMPOUNDING:
            closed_today = [t for t in all_trades
                            if t.get("exit_date", "") == day_str]
            for t in closed_today:
                equity += t["pnl_usd"]
            equity = max(equity, 1)  # never go below $1

        # ── Position size: fixed or equity-based ──
        pos_size = (equity / MAX_POSITIONS) if COMPOUNDING else POSITION_SIZE_USD

        # ── Release expired slots ──
        portfolio = [p for p in portfolio if p["exit_date"] > day + timedelta(days=1)]
        slots_used = len(portfolio)
        slots_free = MAX_POSITIONS - slots_used

        # ── Collect all candidate signals for today ──
        day_longs  = []
        day_shorts = []

        for ticker in WATCHLIST:
            if ticker not in all_history:
                continue
            long_sig, short_sig = compute_signals(ticker, day, all_history[ticker])
            if long_sig  and "long"  in sides:
                # skip if already holding a long on this ticker
                if not (ticker in open_long and
                        (day - open_long[ticker]).days < LONG_HOLD_DAYS):
                    day_longs.append((ticker, long_sig))
            if short_sig and "short" in sides:
                if not (ticker in open_short and
                        (day - open_short[ticker]).days < SHORT_HOLD_DAYS):
                    day_shorts.append((ticker, short_sig))

        # Tag macro events
        is_macro_long  = len(day_longs)  >= MACRO_EVENT_THRESHOLD
        is_macro_short = len(day_shorts) >= MACRO_EVENT_THRESHOLD
        for _, s in day_longs:  s.is_macro_event = is_macro_long
        for _, s in day_shorts: s.is_macro_event = is_macro_short

        # ── Merge and sort ALL candidates by score descending ──
        # Higher score = higher priority for a slot
        all_candidates = [(t, s, "long")  for t, s in day_longs] +                          [(t, s, "short") for t, s in day_shorts]
        all_candidates.sort(key=lambda x: x[1].composite_score, reverse=True)

        day_trades  = []
        slots_taken = 0  # slots consumed this day

        for ticker, signal, direction in all_candidates:
            # Hard cap: never exceed MAX_POSITIONS regardless of timing
            if (slots_used + slots_taken) >= MAX_POSITIONS or slots_taken >= slots_free:
                # No more slots available today - remaining signals skipped
                skipped = len(all_candidates) - all_candidates.index((ticker, signal, direction))
                log.info(f"  [portfolio full] {skipped} signal(s) skipped "
                         f"({slots_used + slots_taken}/{MAX_POSITIONS} slots used)")
                break

            future = all_history[ticker][all_history[ticker].index > pd.Timestamp(day)]
            if future.empty:
                continue

            ep = float(future.iloc[0]["Open"])
            ed = future.index[0].to_pydatetime()

            trade = simulate_trade(ticker, direction, ed, ep, all_history[ticker], signal, pos_size)
            all_trades.append(asdict(trade))
            slots_taken += 1

            # Track open position using the ACTUAL exit date from simulation
            # so slots are released as soon as the real trade closes (SL/TP/time)
            actual_exit = datetime.strptime(trade.exit_date, "%Y-%m-%d") + timedelta(days=1)
            portfolio.append({"exit_date": actual_exit, "ticker": ticker, "direction": direction})

            if direction == "long":
                open_long[ticker]  = day
            else:
                open_short[ticker] = day

            day_trades.append((trade, signal))

        if day_trades:
            macro_tag = ""
            if is_macro_long  and day_longs:  macro_tag = " [MACRO - broad rally]"
            if is_macro_short and day_shorts: macro_tag = " [MACRO - broad sell-off]"
            log.info(f"{day_str}:{macro_tag}")
            for trade, sig in sorted(day_trades, key=lambda x: x[0].direction):
                col   = Fore.GREEN if trade.pnl_usd >= 0 else Fore.RED
                dcol  = Fore.CYAN if trade.direction == "long" else Fore.MAGENTA
                dlbl  = "L" if trade.direction == "long" else "S"
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
        total       = subset["pnl_usd"].sum()
        macro_pnl   = subset[subset["is_macro_event"]]["pnl_usd"].sum()
        specific    = subset[~subset["is_macro_event"]]["pnl_usd"].sum()

        print(f"\n{colour}{'='*65}")
        print(f"  {label}  |  {direction_label} RESULTS")
        print(f"{'='*65}{Style.RESET_ALL}")
        stats = [
            ["Total trades",             len(subset)],
            ["Win rate",                 f"{wr:.1f}%  ({len(wins)}W / {len(losses)}L)"],
            ["Avg return/trade",         f"{subset['pnl_pct'].mean():+.2f}%"],
            ["Avg winning trade",        f"${avg_w:+.2f}"],
            ["Avg losing trade",         f"${avg_l:+.2f}"],
            ["Profit factor",            f"{pf:.2f}"],
            ["Sharpe ratio",             f"{sharpe:.2f}"],
            ["Total P&L",                f"${total:+.2f}"],
            ["  macro event P&L",        f"${macro_pnl:+.2f}"],
            ["  stock-specific P&L",     f"${specific:+.2f}"],
            ["Stopped out",              int((subset["exit_reason"]=="closed_sl").sum())],
            ["Hit take profit",          int((subset["exit_reason"]=="closed_tp").sum())],
            ["Exited by time",           int((subset["exit_reason"]=="closed_time").sum())],
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

    total_wr    = len(df[df["pnl_usd"]>0]) / len(df) * 100
    long_pnl    = longs["pnl_usd"].sum()  if not longs.empty  else 0
    short_pnl   = shorts["pnl_usd"].sum() if not shorts.empty else 0
    macro_pnl   = df[df["is_macro_event"]]["pnl_usd"].sum()
    specific    = df[~df["is_macro_event"]]["pnl_usd"].sum()

    total_return_pct = (long_pnl + short_pnl) / TOTAL_CAPITAL * 100
    print(tabulate([
        ["Total capital",             f"${TOTAL_CAPITAL:,}"],
        ["Mode",                      "Compounding" if COMPOUNDING else "Fixed sizing"],
        ["Max positions (slots)",     MAX_POSITIONS],
        ["Position size",             f"${POSITION_SIZE_USD:.0f}"],
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

    print(f"\n{Fore.CYAN}  Interpretation:{Style.RESET_ALL}")
    if specific > 0:
        print(f"  {Fore.GREEN}+ Stock-specific signals profitable (${specific:+.2f}) - genuine edge beyond market moves{Style.RESET_ALL}")
    else:
        print(f"  {Fore.YELLOW}~ Stock-specific signals flat/negative - edge mainly from macro events{Style.RESET_ALL}")
    if short_pnl > 0:
        print(f"  {Fore.GREEN}+ Short side profitable (${short_pnl:+.2f}) - strategy works in both directions{Style.RESET_ALL}")
    elif short_pnl > -200:
        print(f"  {Fore.YELLOW}~ Short side near breakeven (${short_pnl:+.2f}) - limited downtrends in this period{Style.RESET_ALL}")
    else:
        print(f"  {Fore.RED}- Short side losing (${short_pnl:+.2f}) - may need threshold adjustment{Style.RESET_ALL}")

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
    df["entry_date"] = pd.to_datetime(df["entry_date"])
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
    ax.set_title("Cumulative P&L - Long/Short Strategy", color="#ccc", pad=8)
    ax.set_ylabel("USD", color="#888")
    ax.legend(facecolor="#111", edgecolor="#333", labelcolor="#ccc", fontsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))

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
    out = DATA_DIR / "backtest_ls_chart.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="#0d0d0d")
    plt.show()
    print(f"Chart saved to {out}")

# ── Score segmentation analysis ─────────────────────────────────────────────────────────────

def score_segment_analysis(show_chart: bool = False):
    """
    Groups all trades by composite score bucket and shows average
    win rate, average return %, and trade count for each segment.
    Helps answer: do higher-scoring signals actually perform better?
    """
    if not BT_TRADES_FILE.exists():
        print("Run backtest first.")
        return

    with open(BT_TRADES_FILE, encoding="utf-8") as f:
        trades = json.load(f)
    if not trades:
        print("No trades to analyse.")
        return

    df = pd.DataFrame(trades)

    # Define score buckets
    bins   = [0.0, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.90, 1.01]
    labels = ["0.50-0.55","0.55-0.60","0.60-0.65","0.65-0.70",
              "0.70-0.75","0.75-0.80","0.80-0.90","0.90-1.00"]

    df["score_bucket"] = pd.cut(
        df["composite_score"], bins=bins, labels=labels, right=False
    )

    print(f"\n{Fore.CYAN}{'='*65}")
    print(f"  SCORE SEGMENTATION ANALYSIS")
    print(f"  Does a higher composite score predict better outcomes?")
    print(f"{"="*65}{Style.RESET_ALL}\n")

    # Overall stats for reference
    overall_wr  = (df["pnl_usd"] > 0).mean() * 100
    overall_ret = df["pnl_pct"].mean()
    print(f"  Overall baseline — win rate: {overall_wr:.1f}%  avg return: {overall_ret:+.2f}%\n")

    rows       = []
    chart_data = []

    for label in labels:
        bucket = df[df["score_bucket"] == label]
        if bucket.empty:
            continue

        n       = len(bucket)
        wins    = (bucket["pnl_usd"] > 0).sum()
        wr      = wins / n * 100
        avg_ret = bucket["pnl_pct"].mean()
        avg_win = bucket.loc[bucket["pnl_usd"] > 0, "pnl_usd"].mean()
        avg_los = bucket.loc[bucket["pnl_usd"] <= 0, "pnl_usd"].mean()
        pf      = abs(avg_win / avg_los) if avg_los and avg_los != 0 else 0

        # Direction breakdown
        n_long  = (bucket["direction"] == "long").sum()
        n_short = (bucket["direction"] == "short").sum()

        # Win rate vs baseline indicator
        wr_vs_base = wr - overall_wr
        ret_vs_base = avg_ret - overall_ret
        wr_flag  = f"{Fore.GREEN}+{wr_vs_base:.1f}%{Style.RESET_ALL}" if wr_vs_base >= 0                    else f"{Fore.RED}{wr_vs_base:.1f}%{Style.RESET_ALL}"
        ret_flag = f"{Fore.GREEN}{ret_vs_base:+.2f}%{Style.RESET_ALL}" if ret_vs_base >= 0                    else f"{Fore.RED}{ret_vs_base:+.2f}%{Style.RESET_ALL}"

        rows.append([
            label,
            n,
            f"{n_long}L / {n_short}S",
            f"{wr:.1f}%",
            wr_flag,
            f"{avg_ret:+.2f}%",
            ret_flag,
            f"{pf:.2f}",
        ])
        chart_data.append({
            "bucket": label, "n": n, "wr": wr, "avg_ret": avg_ret, "pf": pf
        })

    print(tabulate(
        rows,
        headers=["Score range","Trades","Dir split","Win rate","vs baseline",
                 "Avg return","vs baseline","Profit factor"],
        tablefmt="simple"
    ))

    # Interpretation
    print(f"\n{Fore.CYAN}{'='*65}")
    print(f"  SCORE SEGMENTATION ANALYSIS")
    print(f"  Does a higher composite score predict better outcomes?")
    print(f"{'='*65}{Style.RESET_ALL}\n")
    if chart_data:
        best_wr  = max(chart_data, key=lambda x: x["wr"])
        best_ret = max(chart_data, key=lambda x: x["avg_ret"])
        worst_wr = min(chart_data, key=lambda x: x["wr"])

        if best_wr["bucket"] == best_ret["bucket"]:
            print(f"  Score band {best_wr['bucket']} dominates — "
                  f"highest win rate ({best_wr['wr']:.1f}%) AND "
                  f"best avg return ({best_ret['avg_ret']:+.2f}%)")
        else:
            print(f"  Best win rate: {best_wr['bucket']} ({best_wr['wr']:.1f}%)")
            print(f"  Best avg return: {best_ret['bucket']} ({best_ret['avg_ret']:+.2f}%)")

        # Check if score is monotonically predictive
        wrs = [d["wr"] for d in chart_data]
        if len(wrs) > 2:
            if wrs[-1] > wrs[0]:
                print(f"  {Fore.GREEN}Higher scores tend to produce better win rates "
                      f"(positive correlation){Style.RESET_ALL}")
            else:
                print(f"  {Fore.YELLOW}Score does not clearly predict win rate — "
                      f"consider recalibrating thresholds{Style.RESET_ALL}")

        low_score_trades = sum(d["n"] for d in chart_data if d["wr"] < overall_wr)
        if low_score_trades > 0:
            low_labels = [d["bucket"] for d in chart_data if d["wr"] < overall_wr]
            print(f"  Bands below baseline: {', '.join(low_labels)} — "
                  f"consider raising LONG_SCORE_MIN to filter these out")

    if show_chart:
        try:
            import matplotlib.pyplot as plt
            import numpy as np
        except ImportError:
            print("pip install matplotlib"); return

        if not chart_data:
            return

        buckets = [d["bucket"] for d in chart_data]
        wrs     = [d["wr"]     for d in chart_data]
        rets    = [d["avg_ret"] for d in chart_data]
        counts  = [d["n"]      for d in chart_data]
        x       = np.arange(len(buckets))
        width   = 0.38

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), facecolor="#0d0d0d")
        for ax in (ax1, ax2):
            ax.set_facecolor("#111")
            ax.tick_params(colors="#888")
            ax.spines[:].set_color("#333")

        # Win rate bars
        bar_cols = ["#1D9E75" if w >= overall_wr else "#D85A30" for w in wrs]
        bars = ax1.bar(x, wrs, color=bar_cols, width=0.6, alpha=0.85)
        ax1.axhline(overall_wr, color="#F5A623", lw=1.5, ls="--",
                    label=f"Overall {overall_wr:.1f}%")
        ax1.set_title("Win Rate by Score Band", color="#ccc", pad=8)
        ax1.set_ylabel("Win Rate %", color="#888")
        ax1.set_xticks(x); ax1.set_xticklabels(buckets, rotation=30, ha="right",
                                                  fontsize=8, color="#888")
        ax1.legend(facecolor="#111", edgecolor="#333", labelcolor="#ccc", fontsize=9)
        # Count labels on bars
        for bar, count in zip(bars, counts):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                     f"n={count}", ha="center", va="bottom", fontsize=7, color="#aaa")

        # Avg return bars
        ret_cols = ["#1D9E75" if r >= 0 else "#D85A30" for r in rets]
        bars2 = ax2.bar(x, rets, color=ret_cols, width=0.6, alpha=0.85)
        ax2.axhline(overall_ret, color="#F5A623", lw=1.5, ls="--",
                    label=f"Overall {overall_ret:+.2f}%")
        ax2.axhline(0, color="#555", lw=0.8, ls=":")
        ax2.set_title("Avg Return % by Score Band", color="#ccc", pad=8)
        ax2.set_ylabel("Avg Return %", color="#888")
        ax2.set_xticks(x); ax2.set_xticklabels(buckets, rotation=30, ha="right",
                                                  fontsize=8, color="#888")
        ax2.legend(facecolor="#111", edgecolor="#333", labelcolor="#ccc", fontsize=9)
        for bar, count in zip(bars2, counts):
            ax2.text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + (0.05 if bar.get_height() >= 0 else -0.15),
                     f"n={count}", ha="center", va="bottom", fontsize=7, color="#aaa")

        plt.suptitle("Score Segmentation Analysis", color="#ccc", fontsize=12, y=1.01)
        plt.tight_layout(pad=2)
        out = DATA_DIR / "score_segmentation.png"
        plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="#0d0d0d")
        plt.show()
        print(f"  Chart saved to {out}")

# ── Open positions histogram ───────────────────────────────────────────────────

def plot_open_positions():
    """
    For each calendar day in the backtest, count how many trades were
    simultaneously open (entry_date <= day <= exit_date) and plot as
    a histogram showing the distribution of position counts.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("pip install matplotlib"); return

    if not BT_TRADES_FILE.exists():
        print("Run backtest first."); return
    with open(BT_TRADES_FILE, encoding="utf-8") as f:
        trades = json.load(f)
    if not trades:
        print("No trades."); return

    df = pd.DataFrame(trades)
    df["entry_date"] = pd.to_datetime(df["entry_date"])
    df["exit_date"]  = pd.to_datetime(df["exit_date"])

    # Build a daily count of open positions
    all_days = pd.date_range(df["entry_date"].min(), df["exit_date"].max(), freq="D")
    daily_counts = {"date": [], "open": [], "longs": [], "shorts": []}

    for day in all_days:
        open_mask  = (df["entry_date"] <= day) & (df["exit_date"] >= day)
        long_mask  = open_mask & (df["direction"] == "long")
        short_mask = open_mask & (df["direction"] == "short")
        daily_counts["date"].append(day)
        daily_counts["open"].append(open_mask.sum())
        daily_counts["longs"].append(long_mask.sum())
        daily_counts["shorts"].append(short_mask.sum())

    counts_df = pd.DataFrame(daily_counts)
    max_open  = int(counts_df["open"].max())
    avg_open  = counts_df["open"].mean()
    peak_day  = counts_df.loc[counts_df["open"].idxmax(), "date"].strftime("%Y-%m-%d")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), facecolor="#0d0d0d")
    for ax in (ax1, ax2):
        ax.set_facecolor("#111"); ax.tick_params(colors="#888"); ax.spines[:].set_color("#333")

    # Left: daily open position count over time (stacked longs/shorts)
    ax1.bar(counts_df["date"], counts_df["longs"],
            color="#1D9E75", alpha=0.8, label="Longs", width=1.0)
    ax1.bar(counts_df["date"], counts_df["shorts"],
            bottom=counts_df["longs"], color="#D85A30", alpha=0.8, label="Shorts", width=1.0)
    ax1.axhline(avg_open, color="#378ADD", lw=1, ls="--", label=f"Avg {avg_open:.1f}")
    ax1.set_title("Open Positions Per Day", color="#ccc", pad=8)
    ax1.set_ylabel("# Positions", color="#888")
    ax1.legend(facecolor="#111", edgecolor="#333", labelcolor="#ccc", fontsize=9)
    ax1.set_xlabel(f"Peak: {max_open} positions on {peak_day}", color="#888")

    # Right: histogram of how often each position count occurred
    bins = range(0, max_open + 2)
    ax2.hist(counts_df["open"], bins=bins, color="#378ADD", alpha=0.85,
             edgecolor="#0d0d0d", linewidth=0.5, align="left")
    ax2.axvline(avg_open, color="#F5A623", lw=1.5, ls="--", label=f"Avg {avg_open:.1f}")
    ax2.set_title("Distribution of Daily Open Position Count", color="#ccc", pad=8)
    ax2.set_xlabel("# Simultaneously Open Positions", color="#888")
    ax2.set_ylabel("# Days", color="#888")
    ax2.legend(facecolor="#111", edgecolor="#333", labelcolor="#ccc", fontsize=9)

    plt.tight_layout(pad=2)
    out = DATA_DIR / "open_positions_histogram.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="#0d0d0d")
    plt.show()
    print(f"Open positions chart saved to {out}")
    print(f"  Peak simultaneous positions : {max_open} (on {peak_day})")
    print(f"  Average open positions/day  : {avg_open:.1f}")
    print(f"  Capital required at peak    : ${max_open * POSITION_SIZE_USD:,} "
          f"({max_open} × ${POSITION_SIZE_USD:,})")

# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    days       = 30
    long_only  = "--long-only"  in sys.argv
    short_only = "--short-only" in sys.argv
    chart      = "--chart"      in sys.argv
    positions  = "--positions"  in sys.argv
    scores     = "--scores"     in sys.argv
    scores_chart = "--scores-chart" in sys.argv
    start_date = None
    end_date   = None

    if "--days" in sys.argv:
        idx  = sys.argv.index("--days")
        days = int(sys.argv[idx + 1])

    if "--start" in sys.argv:
        idx        = sys.argv.index("--start")
        start_date = sys.argv[idx + 1]   # expects YYYY-MM-DD

    if "--end" in sys.argv:
        idx      = sys.argv.index("--end")
        end_date = sys.argv[idx + 1]     # expects YYYY-MM-DD

    trades = run_backtest(
        days_back=days,
        long_only=long_only,
        short_only=short_only,
        start=start_date,
        end=end_date,
    )
    if chart and trades:
        plot_results()
    if positions and trades:
        plot_open_positions()
    if scores or scores_chart:
        score_segment_analysis(show_chart=scores_chart)
