"""
fear_greed.py — CNN Fear & Greed Index fetcher
================================================
Pulls the daily historical Fear & Greed score from CNN's public (but
undocumented) dataviz endpoint and caches it locally so the backtests
don't have to hit the network every run.

The live endpoint only returns reliable daily history from 2021-01-01
onward — earlier start dates 500 error.

Usage:
    python fear_greed.py                # fetch/refresh and print summary
    python fear_greed.py --start 2022-01-01

Output:
    data/fear_greed.json   — [{date, score, rating}, ...]
    data/fear_greed.csv    — same data as CSV
"""

import sys
import json
import urllib.request
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd
from tabulate import tabulate
from colorama import Fore, Style, init

init(autoreset=True)

CNN_URL = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata"
EARLIEST_START = "2021-01-01"  # earlier dates 500 error on CNN's endpoint

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Referer": "https://edition.cnn.com/markets/fear-and-greed",
    "Origin": "https://edition.cnn.com",
    "Accept-Language": "en-US,en;q=0.9",
}

DATA_DIR    = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)
FG_JSON     = DATA_DIR / "fear_greed.json"
FG_CSV      = DATA_DIR / "fear_greed.csv"


def fetch_fear_greed(start: str = EARLIEST_START) -> list:
    """Fetch daily Fear & Greed history from CNN starting at `start` (YYYY-MM-DD)."""
    req = urllib.request.Request(f"{CNN_URL}/{start}", headers=HEADERS)
    with urllib.request.urlopen(req, timeout=15) as resp:
        payload = json.loads(resp.read().decode("utf-8"))

    points = payload["fear_and_greed_historical"]["data"]
    by_date = {}
    for p in points:
        date = datetime.fromtimestamp(p["x"] / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
        by_date[date] = {"date": date, "score": round(p["y"], 2), "rating": p["rating"]}
    return sorted(by_date.values(), key=lambda r: r["date"])


def save(rows: list):
    with open(FG_JSON, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    pd.DataFrame(rows).to_csv(FG_CSV, index=False, encoding="utf-8")


def load_fear_greed() -> pd.DataFrame:
    """Load the cached data as a DataFrame indexed by date (pd.Timestamp), for
    merging into backtests, e.g.:
        fg = load_fear_greed()
        df = df.join(fg, on="date")
    """
    if not FG_JSON.exists():
        raise FileNotFoundError("No cached data — run `python fear_greed.py` first.")
    with open(FG_JSON, encoding="utf-8") as f:
        rows = json.load(f)
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date")


def print_summary(rows: list):
    df = pd.DataFrame(rows)
    latest = rows[-1]
    col = Fore.RED if latest["score"] < 45 else Fore.GREEN if latest["score"] > 55 else Fore.YELLOW
    print(f"\n{Fore.WHITE}{'='*55}")
    print(f"  Fear & Greed Index — {rows[0]['date']} to {rows[-1]['date']}")
    print(f"{'='*55}{Style.RESET_ALL}")
    print(tabulate([
        ["Days cached",   len(df)],
        ["Latest score",  f"{col}{latest['score']:.1f} ({latest['rating']}){Style.RESET_ALL}"],
        ["Mean score",    f"{df['score'].mean():.1f}"],
        ["Min / Max",     f"{df['score'].min():.1f} / {df['score'].max():.1f}"],
        ["Rating counts", {k: int(v) for k, v in df['rating'].value_counts().items()}],
    ], tablefmt="simple", colalign=("left", "left")))
    print(f"\n  Saved to {FG_JSON} and {FG_CSV}\n")


if __name__ == "__main__":
    start = EARLIEST_START
    if "--start" in sys.argv:
        start = sys.argv[sys.argv.index("--start") + 1]

    print(f"Fetching Fear & Greed history from {start}...")
    rows = fetch_fear_greed(start)
    save(rows)
    print_summary(rows)
