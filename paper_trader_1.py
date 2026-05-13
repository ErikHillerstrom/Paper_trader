"""
Informed Trading Signal — Paper Trader v2
==========================================
Two-phase daily workflow matching the backtest model exactly:

  PHASE 1 — Evening scan (run at 22:30 Swedish / 16:30 ET, after US close)
    - Computes signals on today's completed candle
    - Scores and queues the top signals (up to free slots) for tomorrow
    - Does NOT record entry prices yet (market is closed)

  PHASE 2 — Morning open check (run at 15:35 Swedish / 09:35 ET, after US open)
    - Records actual open prices for queued signals → opens paper trades
    - Checks all open trades for stop loss / take profit / time exit
    - Prints current portfolio status

Run modes:
    python paper_trader.py --evening     # Phase 1: signal scan
    python paper_trader.py --morning     # Phase 2: open fills + exit checks
    python paper_trader.py --status      # Print portfolio summary anytime
    python paper_trader.py --schedule    # Run both phases automatically on schedule

Requirements:
    pip install yfinance pandas schedule colorama tabulate
"""

import os, sys, json, time, logging
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional
from tabulate import tabulate
from colorama import Fore, Style, init
init(autoreset=True)

# ── Config — mirror your backtest settings exactly ────────────────────────────

# Signal detection
CALL_PUT_RATIO_MIN   = 3.0
VPIN_THRESHOLD       = 0.65

# Long parameters
LONG_SCORE_MIN       = 0.50
LONG_MIN_SIGNALS     = 2
LONG_HOLD_DAYS       = 3
LONG_STOP_LOSS_PCT   = 0.01
LONG_TAKE_PROFIT_PCT = 0.10

# Short parameters
SHORT_SCORE_MIN      = 0.5
SHORT_MIN_SIGNALS    = 2
SHORT_HOLD_DAYS      = 2
SHORT_STOP_LOSS_PCT  = 0.01
SHORT_TAKE_PROFIT_PCT= 0.10

# Portfolio limits
TOTAL_CAPITAL        = 25_000
MAX_POSITIONS        = 5
COMPOUNDING          = True   # True = position size grows/shrinks with equity
POSITION_SIZE_USD    = TOTAL_CAPITAL / MAX_POSITIONS  # used only when COMPOUNDING=False

# Macro event threshold (flag but still trade)
MACRO_EVENT_THRESHOLD = 8

WATCHLIST = [
    "NVDA","MSFT","AAPL","AMZN","META","GOOGL","TSLA","JPM",
    "XOM","PFE","MRNA","AMD","NFLX","CRM","INTC","BAC","GS",
    "ABBV","LLY","UNH","V","MA","AVGO","ORCL","ADBE",
]

DATA_DIR      = Path("./data")
DATA_DIR.mkdir(exist_ok=True)
TRADES_FILE   = DATA_DIR / "paper_trades.json"
QUEUE_FILE    = DATA_DIR / "signal_queue.json"   # signals waiting for open price
SIGNALS_FILE  = DATA_DIR / "signals_log.json"

# ── Email notifications ───────────────────────────────────────────────────────
# Setup (one-time):
#   1. Enable 2-Step Verification on your Google account.
#   2. Go to myaccount.google.com/apppasswords and create an App Password.
#   3. Set the three environment variables below (or edit the defaults directly).
#      NOTIFY_FROM     — the Gmail address you send FROM
#      NOTIFY_PASSWORD — the App Password (not your normal Gmail password)
#      NOTIFY_TO       — the address to send TO (can be the same Gmail address)

NOTIFY_ENABLED  = True
SMTP_HOST       = "smtp.gmail.com"
SMTP_PORT       = 587
SMTP_USER       = os.environ.get("NOTIFY_FROM",     "")   # sending Gmail address
SMTP_PASSWORD   = os.environ.get("NOTIFY_PASSWORD", "")   # Gmail App Password
NOTIFY_TO       = os.environ.get("NOTIFY_TO",       "")   # recipient (your Gmail)

# Fix Windows console encoding
import io
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "buffer"):
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[
        logging.FileHandler(DATA_DIR / "paper_trader.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

# ── Data helpers ──────────────────────────────────────────────────────────────

def load_json(path: Path) -> list:
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    except Exception:
        return []

def save_json(path: Path, data):
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")

# ── SMS notification ──────────────────────────────────────────────────────────

def send_notification(subject: str, body: str):
    """Send a run-summary email via Gmail SMTP."""
    if not NOTIFY_ENABLED:
        return
    if not all([SMTP_USER, SMTP_PASSWORD, NOTIFY_TO]):
        log.warning("Notification skipped — NOTIFY_FROM / NOTIFY_PASSWORD / NOTIFY_TO not set")
        return
    import smtplib
    from email.mime.text import MIMEText
    try:
        msg = MIMEText(body)
        msg["Subject"] = subject
        msg["From"]    = SMTP_USER
        msg["To"]      = NOTIFY_TO
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as s:
            s.ehlo()
            s.starttls()
            s.login(SMTP_USER, SMTP_PASSWORD)
            s.sendmail(SMTP_USER, [NOTIFY_TO], msg.as_string())
        log.info(f"  Notification sent → {NOTIFY_TO}")
    except Exception as e:
        log.warning(f"  Notification failed: {e}")

# ── yfinance data ─────────────────────────────────────────────────────────────

def get_history(ticker: str, days: int = 60) -> pd.DataFrame:
    """Fetch daily OHLCV history for a ticker."""
    end   = datetime.now()
    start = end - timedelta(days=days + 10)
    try:
        df = yf.download(
            ticker, start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            interval="1d", auto_adjust=True, progress=False
        )
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.index = pd.to_datetime(df.index).tz_localize(None)
        return df.dropna(how="all")
    except Exception as e:
        log.warning(f"  {ticker}: history fetch failed — {e}")
        return pd.DataFrame()

def get_current_price(ticker: str) -> Optional[float]:
    """Get the most recent price (last close or current if market open)."""
    try:
        hist = yf.Ticker(ticker).history(period="1d", interval="1m")
        if not hist.empty:
            return round(float(hist["Close"].iloc[-1]), 4)
        hist = yf.Ticker(ticker).history(period="2d")
        if not hist.empty:
            return round(float(hist["Close"].iloc[-1]), 4)
        return None
    except Exception:
        return None

def get_open_price(ticker: str) -> Optional[float]:
    """Get today's opening price."""
    try:
        hist = yf.Ticker(ticker).history(period="1d")
        if not hist.empty:
            return round(float(hist["Open"].iloc[-1]), 4)
        return None
    except Exception:
        return None

# ── Signal computation (mirrors backtest exactly) ─────────────────────────────

def compute_signals(ticker: str, history: pd.DataFrame):
    """
    Compute long and short signals from the most recent completed candle.
    Returns (long_score, long_triggers, short_score, short_triggers) or None.
    """
    if len(history) < 25:
        return None

    today   = history.iloc[-1]
    prior   = history.iloc[-2]
    avg_vol = history["Volume"].iloc[-21:-1].mean()

    vol_spike   = bool(today["Volume"] > avg_vol * 2.0)
    block       = bool(today["Volume"] > avg_vol * 1.5)
    gap_up      = bool(today["Open"]   > prior["Close"] * 1.01)
    gap_down    = bool(today["Open"]   < prior["Close"] * 0.99)
    close_pct   = (today["Close"] - today["Open"]) / today["Open"]
    recent_high = history["High"].iloc[-20:].max()
    recent_low  = history["Low"].iloc[-20:].min()
    near_high   = bool(today["Close"] > recent_high * 0.97)
    near_low    = bool(today["Close"] < recent_low  * 1.03)

    recent   = history.iloc[-10:]
    up_vol   = recent.loc[recent["Close"] > recent["Open"], "Volume"].sum()
    down_vol = recent.loc[recent["Close"] < recent["Open"], "Volume"].sum()
    total_v  = up_vol + down_vol
    vpin     = round(abs(up_vol - down_vol) / total_v, 3) if total_v > 0 else 0.5
    vpin_bull = vpin >= VPIN_THRESHOLD and up_vol   > down_vol
    vpin_bear = vpin >= VPIN_THRESHOLD and down_vol > up_vol

    # Long signals
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

    # Short signals
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

    long_result  = (long_score,  lt) if long_score  >= LONG_SCORE_MIN  and len(lt) >= LONG_MIN_SIGNALS  else None
    short_result = (short_score, st) if short_score >= SHORT_SCORE_MIN and len(st) >= SHORT_MIN_SIGNALS else None

    return long_result, short_result

# ── Portfolio slot counting ───────────────────────────────────────────────────

def count_open_slots() -> int:
    """Count how many slots are currently occupied by open trades."""
    trades = load_json(TRADES_FILE)
    open_count = sum(1 for t in trades if t["status"] == "open")
    return open_count

# ── Phase 1: Evening scan ─────────────────────────────────────────────────────

def run_evening_scan():
    """
    Scan all tickers after market close.
    Score signals, rank by composite score, queue top N for tomorrow's open.
    """
    log.info("=" * 60)
    log.info(f"EVENING SCAN — {datetime.now().strftime('%Y-%m-%d %H:%M')} (after US close)")
    log.info(f"Capital: ${TOTAL_CAPITAL:,} | Slots: {MAX_POSITIONS} | "
             f"Per position: ${POSITION_SIZE_USD:.0f}")
    log.info("=" * 60)

    open_slots = MAX_POSITIONS - count_open_slots()
    log.info(f"Open slots available: {open_slots}/{MAX_POSITIONS}")

    if open_slots <= 0:
        log.info("Portfolio full — no new signals will be queued")
        return

    # Load existing queue — don't re-queue things already waiting
    existing_queue = load_json(QUEUE_FILE)
    queued_keys = {(q["ticker"], q["direction"]) for q in existing_queue}
    slots_already_queued = len(existing_queue)
    slots_remaining = open_slots - slots_already_queued

    log.info(f"Already queued from prior scans: {slots_already_queued}")
    log.info(f"New slots to fill today: {slots_remaining}\n")

    # Get open trades to avoid doubling up on same ticker+direction
    trades = load_json(TRADES_FILE)
    open_longs  = {t["ticker"] for t in trades if t["status"] == "open" and t["direction"] == "long"}
    open_shorts = {t["ticker"] for t in trades if t["status"] == "open" and t["direction"] == "short"}

    # Scan all tickers
    candidates = []
    signals_log = load_json(SIGNALS_FILE)

    for ticker in WATCHLIST:
        log.info(f"  Scanning {ticker}...")
        hist = get_history(ticker)
        if hist.empty or len(hist) < 25:
            log.warning(f"  {ticker}: insufficient history")
            continue

        result = compute_signals(ticker, hist)
        if result is None:
            continue

        long_result, short_result = result

        if long_result and ticker not in open_longs and (ticker, "long") not in queued_keys:
            score, triggers = long_result
            candidates.append({
                "ticker": ticker, "direction": "long",
                "score": score, "signals": triggers,
                "scan_date": datetime.now().strftime("%Y-%m-%d"),
            })
            signals_log.append({
                "ticker": ticker, "date": datetime.now().strftime("%Y-%m-%d"),
                "direction": "long", "score": score, "signals": triggers,
            })
            log.info(f"  {Fore.GREEN}[L] {ticker}: score={score} sigs={triggers}")

        if short_result and ticker not in open_shorts and (ticker, "short") not in queued_keys:
            score, triggers = short_result
            candidates.append({
                "ticker": ticker, "direction": "short",
                "score": score, "signals": triggers,
                "scan_date": datetime.now().strftime("%Y-%m-%d"),
            })
            signals_log.append({
                "ticker": ticker, "date": datetime.now().strftime("%Y-%m-%d"),
                "direction": "short", "score": score, "signals": triggers,
            })
            log.info(f"  {Fore.MAGENTA}[S] {ticker}: score={score} sigs={triggers}")

        time.sleep(0.3)

    # Tag macro events
    new_longs  = [c for c in candidates if c["direction"] == "long"]
    new_shorts = [c for c in candidates if c["direction"] == "short"]
    is_macro_long  = len(new_longs)  >= MACRO_EVENT_THRESHOLD
    is_macro_short = len(new_shorts) >= MACRO_EVENT_THRESHOLD
    for c in candidates:
        c["is_macro"] = (is_macro_long  if c["direction"] == "long"  else is_macro_short)

    # Sort by score descending — best signals get slots first
    candidates.sort(key=lambda x: x["score"], reverse=True)

    # Take only as many as we have slots for
    new_queued = candidates[:max(slots_remaining, 0)]
    skipped    = len(candidates) - len(new_queued)

    if skipped > 0:
        log.info(f"\n  {skipped} signal(s) skipped — not enough slots "
                 f"({open_slots}/{MAX_POSITIONS} slots free)")

    if is_macro_long:
        log.info(f"  [MACRO EVENT] {len(new_longs)} long signals today — broad market move")
    if is_macro_short:
        log.info(f"  [MACRO EVENT] {len(new_shorts)} short signals today — broad sell-off")

    # Append to queue
    updated_queue = existing_queue + new_queued
    save_json(QUEUE_FILE, updated_queue)
    save_json(SIGNALS_FILE, signals_log)

    log.info(f"\nQueued {len(new_queued)} new signal(s) for tomorrow's open")
    log.info(f"Total pending in queue: {len(updated_queue)}")

    if new_queued:
        rows = [[c["ticker"],
                 c["direction"].upper(),
                 f"{c['score']:.2f}",
                 ", ".join(c["signals"][:2]),
                 "MACRO" if c.get("is_macro") else ""]
                for c in new_queued]
        print()
        print(tabulate(rows,
                       headers=["Ticker","Dir","Score","Signals","Note"],
                       tablefmt="simple"))

    print()
    log.info("ACTION NEEDED: Place market orders at tomorrow's open for the above signals")
    log.info("Then run: python paper_trader.py --morning  (at 15:35 Swedish time)")

    # SMS summary
    date_str     = datetime.now().strftime("%m/%d")
    all_trades   = load_json(TRADES_FILE)
    total_pnl    = sum(t.get("pnl_usd", 0) for t in all_trades if t["status"] != "open")
    total_equity = TOTAL_CAPITAL + total_pnl
    equity_str   = f"Equity ${total_equity:,.0f}"
    if new_queued:
        sig_lines = " | ".join(
            f"{c['ticker']} {c['direction'].upper()} ({c['score']:.2f})"
            for c in new_queued
        )
        macro_tag = " [MACRO]" if (is_macro_long or is_macro_short) else ""
        body = (f"Queued {len(new_queued)}{macro_tag}: {sig_lines}\n"
                f"Slots {MAX_POSITIONS - open_slots + len(new_queued)}/{MAX_POSITIONS} used | {equity_str}")
    else:
        body = f"No new signals queued. Slots {MAX_POSITIONS - open_slots}/{MAX_POSITIONS} used | {equity_str}"
    send_notification(f"Evening scan {date_str}", body)

# ── Phase 2: Morning open ─────────────────────────────────────────────────────

def run_morning_open():
    """
    Run after US market opens (15:35 Swedish / 09:35 ET).
    1. Check all open trades for stop/take profit/time exit at today's prices
    2. Fill queued signals at today's open price
    """
    log.info("=" * 60)
    log.info(f"MORNING OPEN — {datetime.now().strftime('%Y-%m-%d %H:%M')} (after US open)")
    log.info("=" * 60)

    today_str = datetime.now().strftime("%Y-%m-%d")
    trades    = load_json(TRADES_FILE)
    changed   = False

    # Step 1: Check exits on open trades
    log.info("\nChecking open positions...")
    newly_closed = []
    for t in trades:
        if t["status"] != "open":
            continue

        # Gracefully handle old trades missing new fields
        t.setdefault("direction", "long")
        t.setdefault("signals", [])
        t.setdefault("position_usd", POSITION_SIZE_USD)

        current = get_current_price(t["ticker"])
        if current is None:
            log.warning(f"  {t['ticker']}: could not fetch price")
            continue

        entry      = t["entry_price"]
        direction  = t["direction"]
        hold_days  = LONG_HOLD_DAYS if direction == "long" else SHORT_HOLD_DAYS
        entry_dt   = datetime.strptime(t["entry_date"], "%Y-%m-%d")
        days_held  = (datetime.now() - entry_dt).days

        if direction == "long":
            stop_p   = entry * (1 - LONG_STOP_LOSS_PCT)
            target_p = entry * (1 + LONG_TAKE_PROFIT_PCT)
            pnl_pct  = (current - entry) / entry * 100
        else:
            stop_p   = entry * (1 + SHORT_STOP_LOSS_PCT)
            target_p = entry * (1 - SHORT_TAKE_PROFIT_PCT)
            pnl_pct  = (entry - current) / entry * 100

        reason = None
        exit_p = current

        if direction == "long":
            if current <= stop_p:
                reason, exit_p = "closed_sl", stop_p
            elif current >= target_p:
                reason, exit_p = "closed_tp", target_p
            elif days_held >= hold_days:
                reason = "closed_time"
        else:
            if current >= stop_p:
                reason, exit_p = "closed_sl", stop_p
            elif current <= target_p:
                reason, exit_p = "closed_tp", target_p
            elif days_held >= hold_days:
                reason = "closed_time"

        if reason:
            if direction == "long":
                final_pct = (exit_p - entry) / entry * 100
            else:
                final_pct = (entry - exit_p) / entry * 100

            pnl_usd = round(t.get('position_usd', POSITION_SIZE_USD) * final_pct / 100, 2)
            t["status"]     = reason
            t["exit_date"]  = today_str
            t["exit_price"] = round(exit_p, 4)
            t["pnl_pct"]    = round(final_pct, 3)
            t["pnl_usd"]    = pnl_usd
            changed         = True
            newly_closed.append(t)

            col = Fore.GREEN if pnl_usd >= 0 else Fore.RED
            dir_label = "[L]" if direction == "long" else "[S]"
            log.info(f"  {col}{dir_label} CLOSED {t['ticker']} "
                     f"[{reason.replace('closed_','')}] "
                     f"entry=${entry:.2f} current=${exit_p:.2f} "
                     f"P&L={final_pct:+.1f}% (${pnl_usd:+.2f}){Style.RESET_ALL}")
        else:
            if direction == "long":
                live_pct = (current - entry) / entry * 100
            else:
                live_pct = (entry - current) / entry * 100
            col = Fore.GREEN if live_pct >= 0 else Fore.RED
            log.info(f"  [{'L' if direction=='long' else 'S'}] HOLD {t['ticker']:5s} "
                     f"entry=${entry:.2f} now=${current:.2f} "
                     f"{col}{live_pct:+.1f}%{Style.RESET_ALL} "
                     f"({days_held}/{hold_days}d)")

    if changed:
        save_json(TRADES_FILE, trades)

    # Step 2: Fill queued signals at today's open
    queue = load_json(QUEUE_FILE)
    if not queue:
        log.info("\nNo signals queued — nothing to open today")
    else:
        log.info(f"\nFilling {len(queue)} queued signal(s) at today's open price...")
        new_trades    = []
        filled_queue  = []

        # Calculate current equity for compounding
        all_closed    = load_json(TRADES_FILE)
        realised_pnl  = sum(t.get("pnl_usd", 0) for t in all_closed if t["status"] != "open")
        current_equity= max(TOTAL_CAPITAL + realised_pnl, 1)
        pos_size      = (current_equity / MAX_POSITIONS) if COMPOUNDING else POSITION_SIZE_USD
        mode_str      = f"compounding (equity=${current_equity:,.0f})" if COMPOUNDING else "fixed"
        log.info(f"  Position size: ${pos_size:.2f} [{mode_str}]")

        # Re-check slots (some may have freed up from exits above)
        open_count = sum(1 for t in (trades + new_trades) if t["status"] == "open")

        for q in queue:
            if open_count >= MAX_POSITIONS:
                log.info(f"  Portfolio full — {len(queue) - len(filled_queue)} remaining signal(s) re-queued")
                break

            open_price = get_open_price(q["ticker"])
            if open_price is None:
                log.warning(f"  {q['ticker']}: could not fetch open price — keeping in queue")
                filled_queue.append(q)
                continue

            hold = LONG_HOLD_DAYS if q["direction"] == "long" else SHORT_HOLD_DAYS
            exit_target = (datetime.now() + timedelta(days=hold)).strftime("%Y-%m-%d")

            if q["direction"] == "long":
                stop_p   = round(open_price * (1 - LONG_STOP_LOSS_PCT),   4)
                target_p = round(open_price * (1 + LONG_TAKE_PROFIT_PCT), 4)
            else:
                stop_p   = round(open_price * (1 + SHORT_STOP_LOSS_PCT),    4)
                target_p = round(open_price * (1 - SHORT_TAKE_PROFIT_PCT),  4)

            trade = {
                "id":              f"{q['ticker']}_{q['direction']}_{today_str}",
                "ticker":          q["ticker"],
                "direction":       q["direction"],
                "entry_date":      today_str,
                "entry_price":     open_price,
                "position_usd":    pos_size,
                "stop_loss":       stop_p,
                "take_profit":     target_p,
                "target_exit_date":exit_target,
                "composite_score": q["score"],
                "signals":         q["signals"],
                "is_macro":        q.get("is_macro", False),
                "status":          "open",
                "exit_date":       "",
                "exit_price":      0.0,
                "pnl_pct":         0.0,
                "pnl_usd":         0.0,
            }
            new_trades.append(trade)
            open_count += 1

            dcol = Fore.CYAN if q["direction"] == "long" else Fore.MAGENTA
            log.info(f"  {dcol}[{'L' if q['direction']=='long' else 'S'}] OPENED "
                     f"{q['ticker']:5s} @ ${open_price:.2f} | "
                     f"SL=${stop_p:.2f} TP=${target_p:.2f} | "
                     f"score={q['score']:.2f} sigs={q['signals']}{Style.RESET_ALL}")

        # Remaining unfilled (portfolio full) stay in queue for tomorrow
        remaining = queue[len([q for q in queue if q not in filled_queue]):]
        remaining = [q for q in queue if q in filled_queue]  # keep only price-fail ones
        save_json(QUEUE_FILE, remaining)

        all_trades = load_json(TRADES_FILE) + new_trades
        save_json(TRADES_FILE, all_trades)
        log.info(f"\nOpened {len(new_trades)} new paper trade(s)")

    print_portfolio_summary()

    # SMS summary
    date_str  = datetime.now().strftime("%m/%d")
    lines     = []
    if newly_closed:
        for t in newly_closed:
            reason_tag = t["status"].replace("closed_", "").upper()
            lines.append(f"CLOSED {t['ticker']} {t['direction'].upper()} "
                         f"[{reason_tag}] {t['pnl_pct']:+.1f}% ${t['pnl_usd']:+.0f}")
    if new_trades:
        for t in new_trades:
            lines.append(f"OPENED {t['ticker']} {t['direction'].upper()} "
                         f"@${t['entry_price']:.2f}")
    all_trades    = load_json(TRADES_FILE)
    total_pnl     = sum(t.get("pnl_usd", 0) for t in all_trades if t["status"] != "open")
    open_count    = sum(1 for t in all_trades if t["status"] == "open")
    total_equity  = TOTAL_CAPITAL + total_pnl
    lines.append(f"Open {open_count}/{MAX_POSITIONS} | P&L ${total_pnl:+.0f} | Equity ${total_equity:,.0f}")
    body = "\n".join(lines) if lines else "No trades opened or closed today."
    send_notification(f"Morning open {date_str}", body)

# ── Portfolio summary ─────────────────────────────────────────────────────────

def print_portfolio_summary():
    trades  = load_json(TRADES_FILE)
    queue   = load_json(QUEUE_FILE)
    open_t  = [t for t in trades if t["status"] == "open"]
    closed  = [t for t in trades if t["status"] != "open"]
    wins    = [t for t in closed if t.get("pnl_usd", 0) > 0]
    total   = sum(t.get("pnl_usd", 0) for t in closed)
    wr      = len(wins) / len(closed) * 100 if closed else 0
    roc     = total / TOTAL_CAPITAL * 100

    print(f"\n{Fore.CYAN}{'='*60}")
    print(f"  PAPER TRADING SUMMARY — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*60}{Style.RESET_ALL}")

    stats = [
        ["Capital",            f"${TOTAL_CAPITAL:,}"],
        ["Max slots",          MAX_POSITIONS],
        ["Open positions",     f"{len(open_t)}/{MAX_POSITIONS}"],
        ["Pending in queue",   len(queue)],
        ["Closed trades",      len(closed)],
        ["Win rate",           f"{wr:.1f}%  ({len(wins)}W / {len(closed)-len(wins)}L)"],
        ["Total paper P&L",    f"${total:+.2f}"],
        ["Return on capital",  f"{roc:+.2f}%"],
        ["Sizing mode",        "Compounding" if COMPOUNDING else "Fixed"],
    ]
    print(tabulate(stats, tablefmt="simple", colalign=("left","right")))

    if open_t:
        print(f"\n{Fore.YELLOW}  Open positions:{Style.RESET_ALL}")
        rows = []
        for t in open_t:
            # Gracefully handle old trades that lack a direction field
            t.setdefault("direction", "long")
            t.setdefault("signals", [])
            current = get_current_price(t["ticker"])
            if current:
                if t["direction"] == "long":
                    live = (current - t["entry_price"]) / t["entry_price"] * 100
                else:
                    live = (t["entry_price"] - current) / t["entry_price"] * 100
                col = Fore.GREEN if live >= 0 else Fore.RED
                live_str = f"{col}{live:+.1f}%{Style.RESET_ALL}"
            else:
                live_str = "?"
            entry_dt  = datetime.strptime(t["entry_date"], "%Y-%m-%d")
            days_held = (datetime.now() - entry_dt).days
            hold      = LONG_HOLD_DAYS if t["direction"] == "long" else SHORT_HOLD_DAYS
            rows.append([
                t["ticker"],
                t["direction"].upper(),
                t["entry_date"],
                f"${t['entry_price']:.2f}",
                live_str,
                f"{days_held}/{hold}d",
                ", ".join(t["signals"][:2]),
            ])
        print(tabulate(rows,
                       headers=["Ticker","Dir","Entry","Price","Live P&L","Hold","Signals"],
                       tablefmt="simple"))

    if queue:
        print(f"\n{Fore.CYAN}  Queued for next open:{Style.RESET_ALL}")
        rows = [[q["ticker"], q["direction"].upper(),
                 f"{q['score']:.2f}", ", ".join(q["signals"][:2])]
                for q in queue]
        print(tabulate(rows, headers=["Ticker","Dir","Score","Signals"], tablefmt="simple"))

    if closed:
        print(f"\n{Fore.CYAN}  Recent closed trades:{Style.RESET_ALL}")
        rows = []
        for t in sorted(closed, key=lambda x: x.get("exit_date",""), reverse=True)[:10]:
            col = Fore.GREEN if t.get("pnl_usd", 0) >= 0 else Fore.RED
            rows.append([
                t["ticker"],
                t["direction"].upper(),
                t["exit_date"],
                t["status"].replace("closed_","").upper(),
                f"{t.get('pnl_pct',0):+.1f}%",
                f"{col}${t.get('pnl_usd',0):+.2f}{Style.RESET_ALL}",
            ])
        print(tabulate(rows,
                       headers=["Ticker","Dir","Exit","Reason","P&L%","P&L$"],
                       tablefmt="simple"))
    print()

# ── Scheduler ─────────────────────────────────────────────────────────────────

def run_schedule():
    try:
        import schedule
    except ImportError:
        print("Run: pip install schedule")
        sys.exit(1)

    log.info("Scheduler started")
    log.info("  Evening scan : 22:30 Swedish (16:30 ET) — weekdays")
    log.info("  Morning open : 15:35 Swedish (09:35 ET) — weekdays")
    log.info("Press Ctrl+C to stop\n")

    for day in ["monday","tuesday","wednesday","thursday","friday"]:
        getattr(schedule.every(), day).at("22:30").do(run_evening_scan)
        getattr(schedule.every(), day).at("15:35").do(run_morning_open)

    while True:
        schedule.run_pending()
        time.sleep(30)

# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if "--evening"  in sys.argv: run_evening_scan()
    elif "--morning" in sys.argv: run_morning_open()
    elif "--status"  in sys.argv: print_portfolio_summary()
    elif "--schedule" in sys.argv: run_schedule()
    else:
        print(__doc__)
        print("Usage:")
        print("  python paper_trader.py --evening    # after market close (22:30 Swedish)")
        print("  python paper_trader.py --morning    # after market open  (15:35 Swedish)")
        print("  python paper_trader.py --status     # check portfolio anytime")
        print("  python paper_trader.py --schedule   # run on automatic schedule")
