"""
Monte Carlo Walk-Forward Optimiser
====================================
Optimises backtest parameters for capital growth across 5 independent
yearly windows using Monte Carlo random search, then averages the best
parameters from each year and tests them on a sixth out-of-sample year.

Why Monte Carlo over grid search:
  - Grid search over 8+ parameters is computationally prohibitive
    (10 values each = 100 million combinations)
  - Monte Carlo randomly samples the parameter space, finds good regions
    fast, and can be stopped at any time with useful results
  - Walk-forward validation across 5 years catches overfitting that a
    single-window optimisation would miss

Usage:
    python monte_carlo_optimiser.py                  # full run, 200 trials/year
    python monte_carlo_optimiser.py --trials 500     # more trials, better results
    python monte_carlo_optimiser.py --trials 50      # quick test run (~2 min)
    python monte_carlo_optimiser.py --chart          # show results chart

Output:
    data/mc_results.json        — all trial results per year
    data/mc_best_params.json    — best params per year + averaged params
    data/mc_oos_result.json     — out-of-sample test result
"""

import sys, json, random, logging
import pandas as pd
import numpy as np
import yfinance as yf
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from tabulate import tabulate
from colorama import Fore, Style, init
init(autoreset=True)

# ── Paths ──────────────────────────────────────────────────────────────────────

DATA_DIR = Path("./data")
DATA_DIR.mkdir(exist_ok=True)
MC_RESULTS_FILE  = DATA_DIR / "mc_results.json"
MC_BEST_FILE     = DATA_DIR / "mc_best_params.json"
MC_OOS_FILE      = DATA_DIR / "mc_oos_result.json"

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
        logging.FileHandler(DATA_DIR / "mc_optimiser.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

# ── Watchlist ──────────────────────────────────────────────────────────────────

WATCHLIST = [
    "NVDA","MSFT","AAPL","AMZN","META","GOOGL","TSLA","JPM",
    "XOM","PFE","MRNA","AMD","NFLX","CRM","INTC","BAC","GS",
    "ABBV","LLY","UNH","V","MA","AVGO","ORCL","ADBE",
]

# ── Fixed constraints (not optimised — structural choices) ─────────────────────

TOTAL_CAPITAL         = 25_000
MAX_POSITIONS         = 17
MACRO_EVENT_THRESHOLD = 8

# ── Parameter search space ─────────────────────────────────────────────────────
# Each parameter: (min, max, step, type)
# type: "float" | "int"

PARAM_SPACE = {
    "max_positions":         (5,    30,   1,    "int"),
    "long_score_min":        (0.50, 0.80, 0.05, "float"),
    "long_min_signals":      (1,    3,    1,    "int"),
    "long_hold_days":        (2,    6,    1,    "int"),
    "long_stop_loss_pct":    (0.03, 0.10, 0.01, "float"),
    "long_take_profit_pct":  (0.06, 0.20, 0.02, "float"),
    "short_score_min":       (0.60, 0.90, 0.05, "float"),
    "short_min_signals":     (1,    3,    1,    "int"),
    "short_hold_days":       (1,    4,    1,    "int"),
    "short_stop_loss_pct":   (0.03, 0.10, 0.01, "float"),
    "short_take_profit_pct": (0.06, 0.20, 0.02, "float"),
}

# ── Year windows ───────────────────────────────────────────────────────────────

def get_year_windows():
    """
    Returns 6 yearly windows: 5 for optimisation, 1 out-of-sample.
    Uses the 6 most recent complete calendar years.
    Current year 2026, so windows are 2020-2025, OOS = most recent full year.
    """
    current_year = datetime.now().year
    windows = []
    # 5 optimisation years + 1 OOS = 6 years back from last complete year
    last_complete = current_year - 1  # 2025
    for y in range(last_complete - 4, last_complete + 1):  # 2021, 2022, 2023, 2024, 2025
        windows.append({
            "year": y,
            "start": f"{y}-01-01",
            "end":   f"{y}-12-31",
            "label": str(y),
        })
    # OOS = last complete year (2025) — but we use 2020 as OOS to have 5 opt years
    # Actually: opt on 2020-2024, test on 2025
    windows_opt = []
    for y in range(last_complete - 5, last_complete):  # 2020..2024
        windows_opt.append({
            "year": y,
            "start": f"{y}-01-01",
            "end":   f"{y}-12-31",
            "label": str(y),
            "type":  "optimisation",
        })
    oos_window = {
        "year":  last_complete,
        "start": f"{last_complete}-01-01",
        "end":   f"{last_complete}-12-31",
        "label": str(last_complete),
        "type":  "out-of-sample",
    }
    return windows_opt, oos_window

# ── History cache ──────────────────────────────────────────────────────────────

_history_cache = {}

def get_history_for_window(start_str: str, end_str: str) -> dict:
    """
    Download history for the window + 60-day buffer for rolling indicators.
    Cached so each ticker is only downloaded once.
    """
    cache_key = f"{start_str}_{end_str}"
    if cache_key in _history_cache:
        return _history_cache[cache_key]

    buffer_start = (datetime.strptime(start_str, "%Y-%m-%d") - timedelta(days=70)
                    ).strftime("%Y-%m-%d")

    log.info(f"  Downloading history {start_str} to {end_str} (+ 70d buffer)...")
    raw = yf.download(
        WATCHLIST,
        start=buffer_start,
        end=end_str,
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
        except Exception:
            pass

    _history_cache[cache_key] = history
    log.info(f"  Loaded {len(history)} tickers")
    return history

# ── Signal computation (self-contained, no global params) ─────────────────────

def compute_signals_with_params(ticker, date, history, params):
    """Compute signals using explicitly passed parameters."""
    past = history[history.index <= pd.Timestamp(date)].copy()
    if len(past) < 25:
        return None, None

    today   = past.iloc[-1]
    prior   = past.iloc[-2]
    avg_vol = past["Volume"].iloc[-21:-1].mean()
    if avg_vol == 0:
        return None, None

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
    vpin_bull = vpin >= 0.65 and up_vol   > down_vol
    vpin_bear = vpin >= 0.65 and down_vol > up_vol

    lt, ls = [], []
    if close_pct > 0.01 and vol_spike and near_high:
        cp = min(close_pct * 100 * (today["Volume"] / avg_vol), 10.0)
        if cp >= 3.0:
            lt.append("cp_ratio_proxy"); ls.append(min(cp / 6.0, 1.0))
    if vol_spike:
        lt.append("vol_spike"); ls.append(min(today["Volume"] / (avg_vol * 3), 1.0))
    if block and not gap_down:
        lt.append("block_trade"); ls.append(0.7)
    if vpin_bull:
        lt.append("vpin_bullish"); ls.append(vpin)
    if gap_up:
        lt.append("gap_up"); ls.append(0.6)

    st, ss = [], []
    if close_pct < -0.01 and vol_spike and near_low:
        pp = min(abs(close_pct) * 100 * (today["Volume"] / avg_vol), 10.0)
        if pp >= 3.0:
            st.append("put_ratio_proxy"); ss.append(min(pp / 6.0, 1.0))
    if vol_spike and close_pct < 0:
        st.append("vol_spike_down"); ss.append(min(today["Volume"] / (avg_vol * 3), 1.0))
    if block and not gap_up and close_pct < -0.005:
        st.append("block_sell"); ss.append(0.7)
    if vpin_bear:
        st.append("vpin_bearish"); ss.append(vpin)
    if gap_down:
        st.append("gap_down"); ss.append(0.6)

    ls_score = round(sum(ls) / len(ls), 3) if ls else 0.0
    ss_score = round(sum(ss) / len(ss), 3) if ss else 0.0

    long_sig  = (ls_score, lt) if ls_score >= params["long_score_min"]  \
                                and len(lt) >= params["long_min_signals"]  else None
    short_sig = (ss_score, st) if ss_score >= params["short_score_min"] \
                                and len(st) >= params["short_min_signals"] else None

    return long_sig, short_sig

# ── Trade simulation (self-contained) ─────────────────────────────────────────

def simulate_trade_with_params(direction, entry_date, entry_price, history, params):
    """Simulate a trade using explicit parameters. Returns (pnl_pct, exit_reason)."""
    future = history[history.index > pd.Timestamp(entry_date)].copy()
    if future.empty:
        return 0.0, "no_data"

    hold     = params["long_hold_days"]  if direction == "long" else params["short_hold_days"]
    sl_pct   = params["long_stop_loss_pct"]   if direction == "long" else params["short_stop_loss_pct"]
    tp_pct   = params["long_take_profit_pct"] if direction == "long" else params["short_take_profit_pct"]

    if direction == "long":
        stop_p   = entry_price * (1 - sl_pct)
        target_p = entry_price * (1 + tp_pct)
    else:
        stop_p   = entry_price * (1 + sl_pct)
        target_p = entry_price * (1 - tp_pct)

    exit_price  = entry_price
    exit_reason = "closed_time"

    for i, (idx, row) in enumerate(future.iterrows()):
        if i >= hold:
            break
        exit_price = float(row["Close"])

        if direction == "long":
            # Conservative: if both breached same candle, assume stop hit first
            if row["Low"] <= stop_p and row["High"] >= target_p:
                exit_price, exit_reason = stop_p, "closed_sl"; break
            elif row["Low"] <= stop_p:
                exit_price, exit_reason = stop_p,   "closed_sl"; break
            elif row["High"] >= target_p:
                exit_price, exit_reason = target_p, "closed_tp"; break
        else:
            if row["High"] >= stop_p and row["Low"] <= target_p:
                exit_price, exit_reason = stop_p, "closed_sl"; break
            elif row["High"] >= stop_p:
                exit_price, exit_reason = stop_p,   "closed_sl"; break
            elif row["Low"] <= target_p:
                exit_price, exit_reason = target_p, "closed_tp"; break

    if direction == "long":
        pnl_pct = (exit_price - entry_price) / entry_price * 100
    else:
        pnl_pct = (entry_price - exit_price) / entry_price * 100

    return round(pnl_pct, 4), exit_reason

# ── Single backtest run ────────────────────────────────────────────────────────

def run_single_backtest(params: dict, history: dict,
                        start_str: str, end_str: str) -> dict:
    """
    Run a full backtest for the given window using params.
    Returns a metrics dict including total_return_pct (the optimisation target).
    """
    scan_start = datetime.strptime(start_str, "%Y-%m-%d")
    scan_end   = datetime.strptime(end_str,   "%Y-%m-%d")

    if not history:
        return {"total_return_pct": -999, "n_trades": 0}

    sample = next(iter(history))
    trading_days = [
        d.to_pydatetime()
        for d in history[sample].index
        if scan_start <= d.to_pydatetime() <= scan_end
    ]

    if not trading_days:
        return {"total_return_pct": -999, "n_trades": 0}

    max_pos    = params.get("max_positions", MAX_POSITIONS)
    pos_size   = TOTAL_CAPITAL / max_pos
    equity     = TOTAL_CAPITAL
    open_long  = {}
    open_short = {}
    portfolio  = []   # {exit_date, ticker, direction}
    all_pnls   = []

    for day in trading_days:
        # Release expired slots
        portfolio = [p for p in portfolio
                     if p["exit_date"] > day + timedelta(days=1)]
        slots_free = max_pos - len(portfolio)

        if slots_free <= 0:
            continue

        candidates = []
        for ticker in WATCHLIST:
            if ticker not in history:
                continue
            hist = history[ticker]
            long_sig, short_sig = compute_signals_with_params(
                ticker, day, hist, params)

            if long_sig:
                sc, tr = long_sig
                if not (ticker in open_long and
                        (day - open_long[ticker]).days < params["long_hold_days"]):
                    candidates.append((ticker, "long", sc))

            if short_sig:
                sc, tr = short_sig
                if not (ticker in open_short and
                        (day - open_short[ticker]).days < params["short_hold_days"]):
                    candidates.append((ticker, "short", sc))

        # Sort by score, take top N up to free slots
        candidates.sort(key=lambda x: x[2], reverse=True)
        taken = 0

        for ticker, direction, score in candidates:
            if (len(portfolio) + taken) >= max_pos or taken >= slots_free:
                break

            hist   = history[ticker]
            future = hist[hist.index > pd.Timestamp(day)]
            if future.empty:
                continue

            ep = float(future.iloc[0]["Open"])
            ed = future.index[0].to_pydatetime()

            pnl_pct, reason = simulate_trade_with_params(
                direction, ed, ep, hist, params)

            pnl_usd = pos_size * pnl_pct / 100
            all_pnls.append(pnl_usd)
            equity += pnl_usd
            taken  += 1

            hold = params["long_hold_days"] if direction == "long" \
                   else params["short_hold_days"]
            exit_dt = ed + timedelta(days=hold + 1)
            portfolio.append({"exit_date": exit_dt,
                               "ticker": ticker, "direction": direction})

            if direction == "long":
                open_long[ticker]  = day
            else:
                open_short[ticker] = day

    if not all_pnls:
        return {"total_return_pct": -999, "n_trades": 0,
                "win_rate": 0, "sharpe": 0, "equity": TOTAL_CAPITAL}

    total_pnl   = sum(all_pnls)
    n_trades    = len(all_pnls)
    win_rate    = sum(1 for p in all_pnls if p > 0) / n_trades * 100
    total_ret   = total_pnl / TOTAL_CAPITAL * 100
    arr         = pd.Series(all_pnls)
    sharpe      = (arr.mean() / arr.std()) if arr.std() > 0 else 0

    return {
        "total_return_pct": round(total_ret, 4),
        "n_trades":         n_trades,
        "win_rate":         round(win_rate, 2),
        "sharpe":           round(float(sharpe), 4),
        "equity":           round(equity, 2),
    }

# ── Parameter sampling ─────────────────────────────────────────────────────────

def sample_params(rng: random.Random) -> dict:
    """Draw a random parameter set from the search space."""
    params = {}
    for name, (lo, hi, step, ptype) in PARAM_SPACE.items():
        steps  = round((hi - lo) / step)
        choice = rng.randint(0, steps)
        val    = lo + choice * step
        if ptype == "int":
            params[name] = int(round(val))
        else:
            params[name] = round(val, 4)
    return params

def average_params(param_list: list) -> dict:
    """Average a list of parameter dicts, rounding to valid step values."""
    averaged = {}
    for name, (lo, hi, step, ptype) in PARAM_SPACE.items():
        vals = [p[name] for p in param_list]
        mean = sum(vals) / len(vals)
        # Snap to nearest valid step
        snapped = round(round((mean - lo) / step) * step + lo, 4)
        snapped = max(lo, min(hi, snapped))
        averaged[name] = int(round(snapped)) if ptype == "int" else round(snapped, 4)
    return averaged

# ── Monte Carlo optimisation for one year ─────────────────────────────────────

def optimise_year(window: dict, history: dict,
                  n_trials: int, seed: int) -> dict:
    """
    Run n_trials random parameter sets on the given year window.
    Returns the best parameter set found and the top-K results.
    """
    rng = random.Random(seed)
    best_result = None
    best_params = None
    best_score  = -999
    all_results = []

    log.info(f"\n  Optimising {window['label']} — {n_trials} trials...")

    for trial in range(n_trials):
        params = sample_params(rng)
        result = run_single_backtest(
            params, history, window["start"], window["end"])

        # Optimisation objective: total return, but penalise < 10 trades (not enough data)
        score = result["total_return_pct"] if result["n_trades"] >= 10 else -999

        all_results.append({
            "trial":  trial,
            "score":  score,
            "params": params,
            "result": result,
        })

        if score > best_score:
            best_score  = score
            best_params = params
            best_result = result
            log.info(f"    Trial {trial+1:>4}/{n_trials}  "
                     f"NEW BEST: return={score:+.2f}%  "
                     f"wr={result['win_rate']:.1f}%  "
                     f"n={result['n_trades']}  "
                     f"max_pos={params.get('max_positions', MAX_POSITIONS)}")

        elif (trial + 1) % 50 == 0:
            log.info(f"    Trial {trial+1:>4}/{n_trials}  "
                     f"best so far: {best_score:+.2f}%")

    # Sort by score
    all_results.sort(key=lambda x: x["score"], reverse=True)

    return {
        "year":        window["label"],
        "best_params": best_params,
        "best_result": best_result,
        "top10":       all_results[:10],
        "n_trials":    n_trials,
    }

# ── Main optimisation run ──────────────────────────────────────────────────────

def run_optimisation(n_trials: int = 200):
    opt_windows, oos_window = get_year_windows()

    log.info("=" * 65)
    log.info("MONTE CARLO WALK-FORWARD OPTIMISATION")
    log.info(f"Optimisation years : {[w['label'] for w in opt_windows]}")
    log.info(f"Out-of-sample year : {oos_window['label']}")
    log.info(f"Trials per year    : {n_trials}")
    log.info(f"Search space       : {len(PARAM_SPACE)} parameters")
    log.info(f"Watchlist          : {len(WATCHLIST)} tickers")
    log.info("=" * 65)

    yearly_results = []

    for i, window in enumerate(opt_windows):
        log.info(f"\n{'='*65}")
        log.info(f"YEAR {i+1}/5 — {window['label']}")
        log.info(f"{'='*65}")

        history = get_history_for_window(window["start"], window["end"])
        result  = optimise_year(window, history, n_trials=n_trials, seed=42 + i)
        yearly_results.append(result)

        log.info(f"\n  Year {window['label']} best params:")
        for k, v in result["best_params"].items():
            log.info(f"    {k:<28} {v}")
        log.info(f"  Best return: {result['best_result']['total_return_pct']:+.2f}%  "
                 f"win rate: {result['best_result']['win_rate']:.1f}%  "
                 f"trades: {result['best_result']['n_trades']}")

    # ── Average parameters across all 5 years ─────────────────────────────────
    log.info(f"\n{'='*65}")
    log.info("AVERAGING PARAMETERS ACROSS 5 YEARS")
    log.info(f"{'='*65}")

    best_params_list = [r["best_params"] for r in yearly_results]
    averaged_params  = average_params(best_params_list)

    log.info("\n  Per-year best values vs averaged:")
    header = ["Parameter"] + [r["year"] for r in yearly_results] + ["AVERAGED"]
    rows   = []
    for param in PARAM_SPACE:
        row = [param]
        row += [r["best_params"][param] for r in yearly_results]
        row += [averaged_params[param]]
        rows.append(row)
    print(tabulate(rows, headers=header, tablefmt="simple"))

    # Save results
    mc_output = {
        "optimisation_years": [r["year"] for r in yearly_results],
        "oos_year":           oos_window["label"],
        "n_trials_per_year":  n_trials,
        "yearly_results":     yearly_results,
        "averaged_params":    averaged_params,
    }
    with open(MC_RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(mc_output, f, indent=2, default=str)

    with open(MC_BEST_FILE, "w", encoding="utf-8") as f:
        json.dump({"averaged_params": averaged_params,
                   "yearly_best": [{"year": r["year"],
                                    "params": r["best_params"],
                                    "result": r["best_result"]}
                                   for r in yearly_results]}, f, indent=2)

    log.info(f"\n  Results saved to {MC_RESULTS_FILE}")

    # ── Out-of-sample test ─────────────────────────────────────────────────────
    log.info(f"\n{'='*65}")
    log.info(f"OUT-OF-SAMPLE TEST — {oos_window['label']}")
    log.info("Using averaged parameters on unseen data")
    log.info(f"{'='*65}")

    oos_history = get_history_for_window(oos_window["start"], oos_window["end"])
    oos_result  = run_single_backtest(
        averaged_params, oos_history, oos_window["start"], oos_window["end"])

    with open(MC_OOS_FILE, "w", encoding="utf-8") as f:
        json.dump({"year": oos_window["label"],
                   "params": averaged_params,
                   "result": oos_result}, f, indent=2)

    # ── Final summary ──────────────────────────────────────────────────────────
    print_summary(yearly_results, averaged_params, oos_result, oos_window)
    return averaged_params, oos_result

# ── Summary ────────────────────────────────────────────────────────────────────

def print_summary(yearly_results, averaged_params, oos_result, oos_window):
    print(f"\n{Fore.CYAN}{'='*65}")
    print(f"  MONTE CARLO OPTIMISATION SUMMARY")
    print(f"{'='*65}{Style.RESET_ALL}\n")

    # Per-year optimisation results
    print(f"{Fore.CYAN}  Optimisation year results:{Style.RESET_ALL}")
    rows = []
    for r in yearly_results:
        rows.append([
            r["year"],
            r["n_trials"],
            f"{r['best_result']['total_return_pct']:+.2f}%",
            f"{r['best_result']['win_rate']:.1f}%",
            r["best_result"]["n_trades"],
            f"{r['best_result']['sharpe']:.2f}",
        ])
    print(tabulate(rows,
                   headers=["Year","Trials","Best return","Win rate","Trades","Sharpe"],
                   tablefmt="simple"))

    # Averaged parameters
    print(f"\n{Fore.CYAN}  Averaged parameters (used for OOS test):{Style.RESET_ALL}")
    param_rows = [[k, v] for k, v in averaged_params.items()]
    print(tabulate(param_rows, headers=["Parameter","Value"], tablefmt="simple"))

    # OOS result
    oos_col = Fore.GREEN if oos_result["total_return_pct"] > 0 else Fore.RED
    print(f"\n{Fore.CYAN}  Out-of-sample result — {oos_window['label']}:{Style.RESET_ALL}")
    oos_rows = [
        ["Return on capital",  f"{oos_col}{oos_result['total_return_pct']:+.2f}%{Style.RESET_ALL}"],
        ["Win rate",           f"{oos_result['win_rate']:.1f}%"],
        ["Total trades",       oos_result["n_trades"]],
        ["Sharpe ratio",       f"{oos_result['sharpe']:.2f}"],
        ["Final equity",       f"${oos_result['equity']:,.2f}"],
    ]
    print(tabulate(oos_rows, tablefmt="simple", colalign=("left","right")))

    # Interpretation
    opt_returns = [r["best_result"]["total_return_pct"] for r in yearly_results]
    avg_opt_ret = sum(opt_returns) / len(opt_returns)
    oos_ret     = oos_result["total_return_pct"]

    print(f"\n{Fore.CYAN}  Interpretation:{Style.RESET_ALL}")
    print(f"  Avg optimisation return : {avg_opt_ret:+.2f}%")
    print(f"  Out-of-sample return    : {oos_ret:+.2f}%")

    degradation = avg_opt_ret - oos_ret
    if degradation < 5:
        print(f"  {Fore.GREEN}Low degradation ({degradation:+.2f}%) — "
              f"parameters generalise well across market regimes{Style.RESET_ALL}")
    elif degradation < 15:
        print(f"  {Fore.YELLOW}Moderate degradation ({degradation:+.2f}%) — "
              f"some overfitting present but strategy still viable{Style.RESET_ALL}")
    else:
        print(f"  {Fore.RED}High degradation ({degradation:+.2f}%) — "
              f"parameters likely overfit; consider more trials or tighter search space{Style.RESET_ALL}")

    if oos_ret > 0:
        print(f"  {Fore.GREEN}OOS test profitable — averaged parameters have genuine "
              f"out-of-sample edge{Style.RESET_ALL}")
    else:
        print(f"  {Fore.RED}OOS test unprofitable — strategy needs rethinking "
              f"before live deployment{Style.RESET_ALL}")

    print(f"\n  Averaged params saved to: {MC_BEST_FILE}")
    print(f"  Copy them into backtest.py or paper_trader.py to use them.\n")

# ── Chart ──────────────────────────────────────────────────────────────────────

def plot_results():
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("pip install matplotlib"); return

    if not MC_RESULTS_FILE.exists():
        print("Run optimisation first."); return

    with open(MC_RESULTS_FILE, encoding="utf-8") as f:
        data = json.load(f)
    if not MC_OOS_FILE.exists():
        print("OOS result not found."); return
    with open(MC_OOS_FILE, encoding="utf-8") as f:
        oos = json.load(f)

    years    = [r["year"] for r in data["yearly_results"]]
    returns  = [r["best_result"]["total_return_pct"] for r in data["yearly_results"]]
    winrates = [r["best_result"]["win_rate"] for r in data["yearly_results"]]

    oos_ret = oos["result"]["total_return_pct"]
    oos_wr  = oos["result"]["win_rate"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), facecolor="#0d0d0d")
    for ax in axes:
        ax.set_facecolor("#111")
        ax.tick_params(colors="#888")
        ax.spines[:].set_color("#333")

    # Returns per year
    ax = axes[0]
    colors = ["#1D9E75" if r > 0 else "#D85A30" for r in returns]
    bars = ax.bar(years, returns, color=colors, alpha=0.85, width=0.6)
    ax.bar([oos["year"]], [oos_ret],
           color="#378ADD", alpha=0.85, width=0.6, label="OOS")
    ax.axhline(0, color="#555", lw=0.8, ls=":")
    ax.set_title("Best Return per Year (Optimisation + OOS)", color="#ccc", pad=8)
    ax.set_ylabel("Return %", color="#888")
    ax.tick_params(axis="x", colors="#888", rotation=0)
    for bar, ret in zip(bars, returns):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f"{ret:+.1f}%", ha="center", va="bottom", fontsize=8, color="#aaa")
    oos_bar_x = len(years)
    ax.text(oos_bar_x - len(years) + len(years),
            oos_ret + 0.3 if oos_ret >= 0 else oos_ret - 2,
            f"{oos_ret:+.1f}%\n(OOS)", ha="center", va="bottom", fontsize=8, color="#378ADD")
    ax.legend(facecolor="#111", edgecolor="#333", labelcolor="#ccc", fontsize=9)

    # Win rates per year
    ax = axes[1]
    all_wrs    = winrates + [oos_wr]
    all_labels = years + [oos["year"]]
    wr_colors  = ["#378ADD" if l == oos["year"] else
                  ("#1D9E75" if w > 55 else "#D85A30")
                  for l, w in zip(all_labels, all_wrs)]
    ax.bar(all_labels, all_wrs, color=wr_colors, alpha=0.85, width=0.6)
    ax.axhline(50, color="#F5A623", lw=1.5, ls="--", label="50% baseline")
    ax.set_title("Win Rate per Year", color="#ccc", pad=8)
    ax.set_ylabel("Win Rate %", color="#888")
    ax.set_ylim(0, 100)
    ax.legend(facecolor="#111", edgecolor="#333", labelcolor="#ccc", fontsize=9)

    # Parameter heatmap — how stable are params across years?
    ax = axes[2]
    param_names = list(PARAM_SPACE.keys())
    yearly_vals = [[r["best_params"][p] for p in param_names]
                   for r in data["yearly_results"]]
    avg_vals    = [data["averaged_params"][p] for p in param_names]

    # Normalise each param to 0-1 range for display
    norm_vals = []
    for i, (pname, (lo, hi, step, _)) in enumerate(PARAM_SPACE.items()):
        col = [row[i] for row in yearly_vals] + [avg_vals[i]]
        norm = [(v - lo) / (hi - lo) if hi > lo else 0.5 for v in col]
        norm_vals.append(norm)

    norm_arr = np.array(norm_vals)
    im = ax.imshow(norm_arr, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)
    ax.set_yticks(range(len(param_names)))
    ax.set_yticklabels([p.replace("_", " ") for p in param_names],
                       fontsize=7, color="#888")
    ax.set_xticks(range(len(years) + 1))
    ax.set_xticklabels(years + ["AVG"], color="#888", fontsize=8)
    ax.set_title("Parameter Stability Across Years\n(green=high, red=low in range)",
                 color="#ccc", pad=8)

    plt.suptitle("Monte Carlo Walk-Forward Optimisation Results",
                 color="#ccc", fontsize=12, y=1.02)
    plt.tight_layout(pad=2)
    out = DATA_DIR / "mc_optimisation_chart.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="#0d0d0d")
    plt.show()
    print(f"Chart saved to {out}")

# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    n_trials = 200
    chart    = "--chart" in sys.argv

    if "--trials" in sys.argv:
        idx      = sys.argv.index("--trials")
        n_trials = int(sys.argv[idx + 1])

    if "--chart" in sys.argv and not MC_RESULTS_FILE.exists():
        print("No results found — run optimisation first")
        print("  python monte_carlo_optimiser.py --trials 200")
        sys.exit(0)

    if "--chart" in sys.argv and MC_RESULTS_FILE.exists():
        plot_results()
        sys.exit(0)

    averaged_params, oos_result = run_optimisation(n_trials=n_trials)

    if chart:
        plot_results()
