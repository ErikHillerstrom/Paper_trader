"""
Alpaca API test — 5-minute bar fetch
Set ALPACA_API_KEY and ALPACA_API_SECRET as environment variables, then run:
    python alpaca_test.py
"""

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
import os
from datetime import datetime, timedelta

# ── Config ────────────────────────────────────────────────────────────────────
API_KEY    = os.environ.get("ALPACA_API_KEY", "")
API_SECRET = os.environ.get("ALPACA_API_SECRET", "")
TICKER     = "AAPL"
DAYS_BACK  = 365 * 5   # how many calendar days to fetch
# ─────────────────────────────────────────────────────────────────────────────

if not (API_KEY and API_SECRET):
    raise SystemExit("Set ALPACA_API_KEY and ALPACA_API_SECRET environment variables first.")

client = StockHistoricalDataClient(API_KEY, API_SECRET)

end   = datetime.now()
start = end - timedelta(days=DAYS_BACK)

request = StockBarsRequest(
    symbol_or_symbols=TICKER,
    timeframe=TimeFrame(5, TimeFrameUnit.Minute),
    start=start,
    end=end,
    feed="iex",
    limit=None,
)

bars = client.get_stock_bars(request)
df   = bars.df

if TICKER in df.index.get_level_values("symbol"):
    df = df.xs(TICKER, level="symbol")

print(f"Fetched {len(df)} 5-min bars for {TICKER}")
print(f"Earliest bar: {df.index.min()}")
print(f"Latest bar:   {df.index.max()}")
print(df.tail(5).to_string())
