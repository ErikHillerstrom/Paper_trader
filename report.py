"""
report.py — view your paper trading results at any time
Usage:
    python report.py              # full summary
    python report.py --open       # open positions only
    python report.py --signals    # all signals logged
    python report.py --pnl        # P&L chart (requires matplotlib)
"""

import json, sys
from pathlib import Path
from datetime import datetime
from tabulate import tabulate
from colorama import Fore, Style, init
init(autoreset=True)

DATA_DIR     = Path("./data")
TRADES_FILE  = DATA_DIR / "paper_trades.json"
SIGNALS_FILE = DATA_DIR / "signals_log.json"

def load(path):
    if path.exists():
        return json.loads(path.read_text())
    print(f"No data at {path} — run paper_trader.py first")
    return []

def summary():
    trades  = load(TRADES_FILE)
    signals = load(SIGNALS_FILE)
    open_t  = [t for t in trades if t["status"] == "open"]
    closed  = [t for t in trades if t["status"] != "open"]
    wins    = [t for t in closed if t["pnl_usd"] > 0]
    losses  = [t for t in closed if t["pnl_usd"] <= 0]
    total   = sum(t["pnl_usd"] for t in closed)
    avg_win = sum(t["pnl_usd"] for t in wins) / len(wins) if wins else 0
    avg_los = sum(t["pnl_usd"] for t in losses) / len(losses) if losses else 0
    wr      = len(wins)/len(closed)*100 if closed else 0
    profit_factor = abs(avg_win / avg_los) if avg_los else 0

    print(f"\n{Fore.CYAN}{'═'*65}")
    print(f"  INFORMED TRADER — PAPER TRADING REPORT")
    print(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'═'*65}{Style.RESET_ALL}\n")

    stats = [
        ["Total signals scanned", len(signals)],
        ["Signals above threshold (trades opened)", len(trades)],
        ["Currently open positions", len(open_t)],
        ["Closed trades", len(closed)],
        ["Win rate", f"{wr:.1f}%  ({len(wins)} W / {len(losses)} L)"],
        ["Avg winning trade", f"${avg_win:+.2f}"],
        ["Avg losing trade",  f"${avg_los:+.2f}"],
        ["Profit factor", f"{profit_factor:.2f}"],
        ["Total paper P&L", f"${total:+.2f}"],
    ]
    print(tabulate(stats, tablefmt="simple", colalign=("left","right")))

    if open_t:
        print(f"\n{Fore.YELLOW}  ── Open Positions ──{Style.RESET_ALL}")
        rows = [[t["ticker"], t["entry_date"], f"${t['entry_price']:.2f}",
                 f"${t['stop_loss']:.2f}", f"${t['take_profit']:.2f}",
                 t["target_exit_date"], ", ".join(t["signals"][:2])]
                for t in open_t]
        print(tabulate(rows, headers=["Ticker","Entry","Price","Stop","Target","Exit By","Signals"], tablefmt="simple"))

    if closed:
        print(f"\n{Fore.CYAN}  ── Closed Trades (most recent 20) ──{Style.RESET_ALL}")
        rows = []
        for t in sorted(closed, key=lambda x: x["exit_date"], reverse=True)[:20]:
            colour = Fore.GREEN if t["pnl_usd"] >= 0 else Fore.RED
            rows.append([
                t["ticker"], t["entry_date"], t["exit_date"],
                t["status"].replace("closed_","").upper(),
                f"{t['pnl_pct']:+.1f}%",
                f"{colour}${t['pnl_usd']:+.2f}{Style.RESET_ALL}",
                f"{t['composite_score']:.2f}",
            ])
        print(tabulate(rows, headers=["Ticker","Entry","Exit","Reason","P&L%","P&L$","Score"], tablefmt="simple"))

    if signals:
        from collections import Counter
        all_sigs = [s for t in signals for s in t["signals_triggered"]]
        sig_counts = Counter(all_sigs)
        print(f"\n{Fore.CYAN}  ── Most Triggered Signals ──{Style.RESET_ALL}")
        print(tabulate(sig_counts.most_common(), headers=["Signal","Count"], tablefmt="simple"))

def show_signals():
    signals = load(SIGNALS_FILE)
    flagged = [s for s in signals if s["composite_score"] >= 0.5]
    print(f"\n{Fore.CYAN}  Signals above threshold ({len(flagged)} of {len(signals)} scanned){Style.RESET_ALL}\n")
    rows = [[s["ticker"], s["date"], f"{s['composite_score']:.2f}",
             f"{s['options_vol_ratio']:.1f}×", f"{s['call_put_ratio']:.1f}",
             "✓" if s["block_detected"] else "—",
             f"{s['vpin_score']:.2f}",
             ", ".join(s["signals_triggered"])]
            for s in sorted(flagged, key=lambda x: x["composite_score"], reverse=True)[:30]]
    print(tabulate(rows, headers=["Ticker","Date","Score","OptVol","C/P","Block","VPIN","Signals"], tablefmt="simple"))

def pnl_chart():
    try:
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        import pandas as pd
    except ImportError:
        print("Run: pip install matplotlib pandas")
        return

    trades = load(TRADES_FILE)
    closed = sorted([t for t in trades if t["status"] != "open"], key=lambda x: x["exit_date"])
    if not closed:
        print("No closed trades yet.")
        return

    df = pd.DataFrame(closed)
    df["exit_date"] = pd.to_datetime(df["exit_date"])
    df["cumulative_pnl"] = df["pnl_usd"].cumsum()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), facecolor="#0d0d0d")
    for ax in (ax1, ax2):
        ax.set_facecolor("#111111")
        ax.tick_params(colors="#888")
        ax.spines[:].set_color("#333")

    ax1.plot(df["exit_date"], df["cumulative_pnl"], color="#378ADD", linewidth=2)
    ax1.axhline(0, color="#555", linewidth=0.8, linestyle="--")
    ax1.fill_between(df["exit_date"], df["cumulative_pnl"], 0,
                     where=df["cumulative_pnl"] >= 0, alpha=0.15, color="#1D9E75")
    ax1.fill_between(df["exit_date"], df["cumulative_pnl"], 0,
                     where=df["cumulative_pnl"] < 0, alpha=0.15, color="#D85A30")
    ax1.set_title("Cumulative Paper P&L", color="#ccc", pad=10)
    ax1.set_ylabel("USD", color="#888")
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))

    colors = ["#1D9E75" if p >= 0 else "#D85A30" for p in df["pnl_usd"]]
    ax2.bar(df["exit_date"], df["pnl_usd"], color=colors, width=0.8)
    ax2.axhline(0, color="#555", linewidth=0.8)
    ax2.set_title("Per-Trade P&L", color="#ccc", pad=10)
    ax2.set_ylabel("USD", color="#888")
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))

    plt.tight_layout(pad=2)
    plt.savefig(DATA_DIR / "pnl_chart.png", dpi=150, bbox_inches="tight", facecolor="#0d0d0d")
    plt.show()
    print(f"Chart saved to {DATA_DIR / 'pnl_chart.png'}")

if __name__ == "__main__":
    if "--signals" in sys.argv:
        show_signals()
    elif "--pnl" in sys.argv:
        pnl_chart()
    else:
        summary()
