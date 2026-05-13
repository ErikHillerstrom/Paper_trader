# Informed Trading Paper Trader

Detects unusual options activity and block trades that may indicate informed
trading ahead of material announcements, then paper-trades those signals to
validate the strategy — no real money involved.

---

## Quick Start

### 1. Install dependencies
```bash
pip install requests pandas schedule colorama tabulate matplotlib
```

### 2. Get free API keys
| Service | URL | Cost | Used for |
|---------|-----|------|----------|
| Polygon.io | https://polygon.io | Free (15-min delay) | Options & stock data |
| Alpaca | https://alpaca.markets | Free | Price quotes (paper account) |

### 3. Set your API keys
Either edit the constants at the top of `paper_trader.py`, or set environment variables:
```bash
export POLYGON_API_KEY="your_key_here"
export ALPACA_API_KEY="your_key_here"
export ALPACA_SECRET_KEY="your_secret_here"
```

### 4. Run a scan right now
```bash
python paper_trader.py --once
```

### 5. Run on a schedule (weekdays at 09:45 + 16:30)
```bash
python paper_trader.py --schedule
```

### 6. View results anytime
```bash
python report.py              # full P&L summary
python report.py --signals    # all signals logged
python report.py --pnl        # matplotlib chart
```

---

## How it works

```
Each day:
  For each ticker in WATCHLIST:
    1. Fetch options snapshot (Polygon)      → vol ratio, C/P ratio, DTE
    2. Scan for block trades                 → single prints > X% of ADV
    3. Estimate VPIN from 5-min bars         → order flow imbalance proxy
    4. Compute composite score (0–1)
    5. If score ≥ threshold AND ≥2 signals:
       → Open a paper trade (logged to data/paper_trades.json)

  For each open paper trade:
    → Check current price
    → Close if: stop loss hit | take profit hit | hold period expired
    → Log outcome to data/paper_trades.json
```

---

## Output files

| File | Contents |
|------|----------|
| `data/paper_trades.json` | All virtual trades with entry/exit/P&L |
| `data/signals_log.json`  | Every signal scored (above and below threshold) |
| `data/pnl_summary.csv`   | Closed trades as CSV for Excel/Sheets |
| `data/trader.log`        | Full scan log |

---

## Tuning the strategy

Edit the constants at the top of `paper_trader.py`:

```python
OPTIONS_VOL_MULTIPLIER = 5      # raise for fewer, stronger signals
CALL_PUT_RATIO_MIN     = 3.0    # raise to filter for more extreme call skew
MAX_DTE                = 14     # lower = more urgent near-term options only
COMPOSITE_SCORE_MIN    = 0.50   # raise for higher confidence trades only
HOLD_DAYS              = 3      # exit after N days
STOP_LOSS_PCT          = -0.05  # -5% stop
TAKE_PROFIT_PCT        = 0.10   # +10% target
```

After 4–6 weeks of paper trading, compare:
- Win rate vs 50% baseline
- Avg return vs transaction costs (~$0.005/share)
- Sharpe ratio vs SPY buy-and-hold

If win rate > 58% with >30 trades, the signal has statistical edge worth
exploring with real capital.

---

## Free tier limitations (Polygon.io)

- Data is 15 minutes delayed — fine for end-of-day analysis
- Options snapshot limited to 250 contracts per call
- VPIN proxy uses stock bars, not true tick data — real VPIN needs a paid plan

Upgrade to Polygon Starter ($29/mo) for real-time data when you're ready.

---

## Legal note

This system only uses publicly observable market data (prices, volumes,
options activity). It does not access or use material non-public information.
Paper trading carries no regulatory risk. Before live trading, review FINRA
and SEC regulations around algorithmic trading in your jurisdiction.
