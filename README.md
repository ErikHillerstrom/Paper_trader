# Informed Trading Paper Trader

Detects unusual volume, order-flow imbalance, and gap events on a 25-ticker
watchlist of US large-caps, scores them into long and short signals, and
paper-trades those signals to validate strategy edge — no real money involved.

---

## Project structure

| File | Purpose |
|------|---------|
| `paper_trader_1.py` | **Main script.** Two-phase live paper trader (evening scan + morning open) |
| `backtest_6.py` | **Latest backtest.** Long/short replay with score segmentation analysis |
| `monte_carlo_optimiser.py` | Walk-forward Monte Carlo parameter optimiser (5 in-sample years + OOS test) |
| `report.py` | P&L summary, open positions, and matplotlib chart viewer |
| `paper_trader.py` | v1 paper trader (single-phase, kept for reference) |
| `backtest.py` / `backtest_4.py` / `backtest_5.py` | Earlier backtest iterations (kept for reference) |

---

## How it works

### Signal detection
Each ticker is scored daily using four publicly observable indicators:

| Signal | Description |
|--------|-------------|
| `vol_spike` | Volume > 2× 20-day average |
| `block_trade` | Volume > 1.5× average (large single-print proxy) |
| `cp_ratio_proxy` / `put_ratio_proxy` | Close% × volume ratio above threshold |
| `vpin_bullish` / `vpin_bearish` | 10-day directional volume imbalance ≥ 0.65 |
| `gap_up` / `gap_down` | Open > prior close by 1% |

A composite score (0–1) is computed as the average of triggered signal strengths.
Trades are opened when score ≥ threshold **and** ≥ 2 signals fire simultaneously.

### Two-phase daily workflow (`paper_trader_1.py`)

```
PHASE 1 — Evening scan (22:30 Swedish / 16:30 ET, after US close)
  → Scores all tickers on today's completed candle
  → Ranks candidates by composite score
  → Queues top N signals (up to free portfolio slots) for tomorrow's open

PHASE 2 — Morning open (15:35 Swedish / 09:35 ET, after US open)
  → Fills queued signals at today's actual open price
  → Checks all open trades for stop loss / take profit / time exit
  → Prints portfolio summary
  → Sends email notification with run summary
```

---

## Quick start

### 1. Install dependencies
```bash
pip install yfinance pandas schedule colorama tabulate matplotlib
```

### 2. Run the paper trader

```bash
# Phase 1 — after US market close
python paper_trader_1.py --evening

# Phase 2 — after US market open
python paper_trader_1.py --morning

# Check portfolio anytime
python paper_trader_1.py --status

# Run both phases automatically on schedule (weekdays)
python paper_trader_1.py --schedule
```

### 3. View results

```bash
python report.py              # full P&L summary
python report.py --open       # open positions only
python report.py --signals    # all signals above threshold
python report.py --pnl        # matplotlib cumulative P&L chart
```

---

## Email notifications

After each phase run the script emails a summary to your inbox.

### Setup (one-time)
1. Enable 2-Step Verification on your Google account
2. Create an App Password at `myaccount.google.com/apppasswords`
3. Set three environment variables (add to your shell profile or Windows user environment):

```bash
NOTIFY_FROM      = "you@gmail.com"       # sending Gmail address
NOTIFY_PASSWORD  = "xxxx xxxx xxxx xxxx" # 16-character App Password
NOTIFY_TO        = "you@gmail.com"       # recipient (can be the same address)
```

**Windows (persistent):**
```powershell
[System.Environment]::SetEnvironmentVariable("NOTIFY_FROM",     "you@gmail.com", "User")
[System.Environment]::SetEnvironmentVariable("NOTIFY_PASSWORD", "xxxx xxxx xxxx xxxx", "User")
[System.Environment]::SetEnvironmentVariable("NOTIFY_TO",       "you@gmail.com", "User")
```

To disable notifications without removing the env vars, set `NOTIFY_ENABLED = False` at the top of `paper_trader_1.py`.

---

## Backtesting

### Run the latest backtest
```bash
python backtest_6.py                        # last 30 days, long + short
python backtest_6.py --days 90              # last 90 days
python backtest_6.py --long-only            # longs only
python backtest_6.py --short-only           # shorts only
python backtest_6.py --start 2024-01-01 --end 2024-12-31   # fixed window
python backtest_6.py --chart                # matplotlib P&L chart
python backtest_6.py --scores               # score segmentation analysis
python backtest_6.py --scores-chart         # score segmentation chart
python backtest_6.py --positions            # open position count histogram
```

### Score segmentation analysis
Shows whether higher composite scores actually predict better outcomes —
useful for deciding whether to raise `LONG_SCORE_MIN` / `SHORT_SCORE_MIN`.

---

## Monte Carlo parameter optimisation

Optimises 11 strategy parameters across 5 independent yearly windows (2020–2024),
averages the per-year best parameters, then tests them on a sixth out-of-sample year (2025).

```bash
python monte_carlo_optimiser.py               # 200 trials/year (~15 min)
python monte_carlo_optimiser.py --trials 500  # more thorough (~40 min)
python monte_carlo_optimiser.py --trials 50   # quick test (~2 min)
python monte_carlo_optimiser.py --chart       # visualise results after a run
```

Parameters optimised:

| Parameter | Search range |
|-----------|-------------|
| `max_positions` | 5 – 30 |
| `long_score_min` | 0.50 – 0.80 |
| `long_min_signals` | 1 – 3 |
| `long_hold_days` | 2 – 6 |
| `long_stop_loss_pct` | 3% – 10% |
| `long_take_profit_pct` | 6% – 20% |
| `short_score_min` | 0.60 – 0.90 |
| `short_min_signals` | 1 – 3 |
| `short_hold_days` | 1 – 4 |
| `short_stop_loss_pct` | 3% – 10% |
| `short_take_profit_pct` | 6% – 20% |

After a run, copy the averaged parameters from `data/mc_best_params.json` into
`paper_trader_1.py` and `backtest_6.py`.

---

## Tuning the strategy manually

Edit the constants at the top of `paper_trader_1.py`:

```python
CALL_PUT_RATIO_MIN    = 3.0    # raise for stricter volume/close confirmation
VPIN_THRESHOLD        = 0.65   # raise for stronger order flow confirmation

LONG_SCORE_MIN        = 0.50   # raise for fewer, higher-confidence longs
LONG_MIN_SIGNALS      = 2      # minimum signals required to open a long
LONG_HOLD_DAYS        = 3      # exit after N days if no SL/TP hit
LONG_STOP_LOSS_PCT    = 0.01   # 1% stop loss
LONG_TAKE_PROFIT_PCT  = 0.10   # 10% take profit

SHORT_SCORE_MIN       = 0.50
SHORT_HOLD_DAYS       = 2      # shorts held shorter — can reverse fast
SHORT_STOP_LOSS_PCT   = 0.01
SHORT_TAKE_PROFIT_PCT = 0.10

TOTAL_CAPITAL         = 25_000
MAX_POSITIONS         = 5
COMPOUNDING           = True   # position size grows/shrinks with equity
```

---

## Output files

| File | Contents |
|------|---------|
| `data/paper_trades.json` | All paper trades with entry / exit / P&L |
| `data/signal_queue.json` | Signals queued for next morning open |
| `data/signals_log.json` | Every signal scored above threshold |
| `data/paper_trader.log` | Full scan log |
| `data/backtest_trades_ls.json` | Backtest trade-by-trade results |
| `data/backtest_summary_ls.csv` | Backtest results as CSV |
| `data/mc_best_params.json` | Best averaged parameters from Monte Carlo run |
| `data/mc_oos_result.json` | Out-of-sample test result |
| `data/mc_results.json` | Full Monte Carlo trial data |

---

## Watchlist

```
NVDA MSFT AAPL AMZN META GOOGL TSLA JPM XOM PFE
MRNA AMD NFLX CRM INTC BAC GS ABBV LLY UNH
V MA AVGO ORCL ADBE
```

---

## When to consider live trading

After 4–6 weeks of paper trading, compare:
- Win rate vs 50% baseline (random)
- Avg return vs estimated transaction costs (~$0.005/share)
- Sharpe ratio vs SPY buy-and-hold
- OOS Monte Carlo result > 0%

A win rate above ~58% with 30+ closed trades suggests statistical edge worth
exploring with small real capital.

---

## Legal note

This system uses only publicly observable market data (prices, volumes).
It does not access or use material non-public information.
Paper trading carries no regulatory risk. Before live trading, review FINRA
and SEC regulations around algorithmic trading in your jurisdiction.
