"""
Paper Trader Dashboard
======================
Lightweight Flask web app displaying live paper trader data.
Run: python dashboard.py
Access: http://<EC2-PUBLIC-IP>:5000
"""

import json
from pathlib import Path
from datetime import datetime
from flask import Flask, render_template_string

app = Flask(__name__)

BASE_DIR    = Path(__file__).parent
DATA_DIR    = BASE_DIR / "data"
TRADES_FILE = DATA_DIR / "paper_trades.json"
QUEUE_FILE  = DATA_DIR / "signal_queue.json"
SIGNALS_FILE= DATA_DIR / "signals_log.json"
LOG_FILE    = BASE_DIR / "cron.log"

TOTAL_CAPITAL = 25_000

def load_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return []

def load_log(n=80):
    try:
        lines = LOG_FILE.read_text(errors="replace").splitlines()
        return lines[-n:]
    except Exception:
        return ["(log file not found)"]

TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta http-equiv="refresh" content="60">
  <title>Paper Trader Dashboard</title>
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css">
  <style>
    body { background: #0d1117; color: #e6edf3; }
    .card { background: #161b22; border: 1px solid #30363d; }
    .card-header { background: #21262d; border-bottom: 1px solid #30363d; font-weight: 600; color: #e6edf3; }
    table { font-size: 0.875rem; }
    th { color: #cdd5df; font-weight: 500; }
    .badge-long  { background: #1f6feb; }
    .badge-short { background: #6e40c9; }
    .pos { color: #3fb950; font-weight: 600; }
    .neg { color: #f85149; font-weight: 600; }
    .tag { font-size: 0.7rem; background: #21262d; border: 1px solid #30363d;
           border-radius: 4px; padding: 1px 5px; margin: 1px; display: inline-block; }
    pre { background: #0d1117; color: #cdd5df; font-size: 0.75rem;
          max-height: 380px; overflow-y: auto; border-radius: 6px;
          padding: 12px; border: 1px solid #30363d; }
    .stat-label { font-size: 0.75rem; color: #cdd5df; }
    .stat-value { font-size: 1.5rem; font-weight: 700; }
    .navbar-brand { font-weight: 700; letter-spacing: .5px; }
    .updated { font-size: 0.75rem; color: #cdd5df; }
  </style>
</head>
<body>
<nav class="navbar navbar-dark" style="background:#161b22; border-bottom:1px solid #30363d;">
  <div class="container-fluid">
    <span class="navbar-brand">Paper Trader</span>
    <span class="updated">Auto-refreshes every 60s &nbsp;|&nbsp; {{ now }}</span>
  </div>
</nav>

<div class="container-fluid py-3">

  <!-- Summary row -->
  <div class="row g-3 mb-3">
    <div class="col-6 col-md-3">
      <div class="card p-3 text-center">
        <div class="stat-label">Total Equity</div>
        <div class="stat-value {% if equity >= total_capital %}pos{% else %}neg{% endif %}">${{ '{:,.0f}'.format(equity) }}</div>
      </div>
    </div>
    <div class="col-6 col-md-3">
      <div class="card p-3 text-center">
        <div class="stat-label">Total P&amp;L</div>
        <div class="stat-value {% if total_pnl >= 0 %}pos{% else %}neg{% endif %}">${{ '{:+,.2f}'.format(total_pnl) }}</div>
      </div>
    </div>
    <div class="col-6 col-md-3">
      <div class="card p-3 text-center">
        <div class="stat-label">Open Positions</div>
        <div class="stat-value" style="color:#58a6ff">{{ open_count }}</div>
      </div>
    </div>
    <div class="col-6 col-md-3">
      <div class="card p-3 text-center">
        <div class="stat-label">Closed Trades</div>
        <div class="stat-value" style="color:#ffffff">{{ closed_count }}</div>
      </div>
    </div>
  </div>

  <div class="row g-3">

    <!-- Open positions -->
    <div class="col-12 col-lg-6">
      <div class="card">
        <div class="card-header">Open Positions ({{ open_count }})</div>
        <div class="card-body p-0">
          {% if open_trades %}
          <table class="table table-dark table-sm mb-0">
            <thead><tr>
              <th>Ticker</th><th>Dir</th><th>Entry</th>
              <th>Date</th><th>Score</th><th>Signals</th>
            </tr></thead>
            <tbody>
            {% for t in open_trades %}
            <tr>
              <td><strong><a href="https://finance.yahoo.com/quote/{{ t.ticker }}" target="_blank" style="color:#58a6ff;text-decoration:none;">{{ t.ticker }}</a></strong></td>
              <td>
                {% if t.direction == 'long' %}
                  <span class="badge badge-long">LONG</span>
                {% else %}
                  <span class="badge badge-short">SHORT</span>
                {% endif %}
              </td>
              <td>${{ '{:.2f}'.format(t.entry_price) }}</td>
              <td>{{ t.entry_date }}</td>
              <td>{{ '{:.2f}'.format(t.score) }}</td>
              <td>{% for s in t.signals %}<span class="tag">{{ s }}</span>{% endfor %}</td>
            </tr>
            {% endfor %}
            </tbody>
          </table>
          {% else %}
          <p class="p-3 mb-0" style="color:#cdd5df">No open positions.</p>
          {% endif %}
        </div>
      </div>
    </div>

    <!-- Signal queue -->
    <div class="col-12 col-lg-6">
      <div class="card">
        <div class="card-header">Signal Queue — Tomorrow's Open ({{ queue|length }})</div>
        <div class="card-body p-0">
          {% if queue %}
          <table class="table table-dark table-sm mb-0">
            <thead><tr>
              <th>Ticker</th><th>Dir</th><th>Score</th><th>Signals</th>
            </tr></thead>
            <tbody>
            {% for q in queue %}
            <tr>
              <td><strong>{{ q.ticker }}</strong></td>
              <td>
                {% if q.direction == 'long' %}
                  <span class="badge badge-long">LONG</span>
                {% else %}
                  <span class="badge badge-short">SHORT</span>
                {% endif %}
              </td>
              <td>{{ '{:.2f}'.format(q.score) }}</td>
              <td>{% for s in q.signals %}<span class="tag">{{ s }}</span>{% endfor %}</td>
            </tr>
            {% endfor %}
            </tbody>
          </table>
          {% else %}
          <p class="p-3 mb-0" style="color:#cdd5df">Queue is empty (evening scan hasn't run yet today).</p>
          {% endif %}
        </div>
      </div>
    </div>

    <!-- Recent closed trades -->
    <div class="col-12">
      <div class="card">
        <div class="card-header">Recent Closed Trades (last 20)</div>
        <div class="card-body p-0">
          {% if closed_trades %}
          <table class="table table-dark table-sm mb-0">
            <thead><tr>
              <th>Ticker</th><th>Dir</th><th>Entry</th><th>Exit</th>
              <th>Entry Date</th><th>Exit Date</th><th>P&amp;L %</th><th>P&amp;L $</th><th>Reason</th>
            </tr></thead>
            <tbody>
            {% for t in closed_trades %}
            <tr>
              <td><strong><a href="https://finance.yahoo.com/quote/{{ t.ticker }}" target="_blank" style="color:#58a6ff;text-decoration:none;">{{ t.ticker }}</a></strong></td>
              <td>
                {% if t.direction == 'long' %}
                  <span class="badge badge-long">LONG</span>
                {% else %}
                  <span class="badge badge-short">SHORT</span>
                {% endif %}
              </td>
              <td>${{ '{:.2f}'.format(t.entry_price) }}</td>
              <td>${{ '{:.2f}'.format(t.exit_price) }}</td>
              <td>{{ t.entry_date }}</td>
              <td>{{ t.exit_date }}</td>
              <td class="{% if t.pnl_pct >= 0 %}pos{% else %}neg{% endif %}">
                {{ '{:+.2f}'.format(t.pnl_pct) }}%
              </td>
              <td class="{% if t.pnl_usd >= 0 %}pos{% else %}neg{% endif %}">
                ${{ '{:+.2f}'.format(t.pnl_usd) }}
              </td>
              <td><span class="tag">{{ t.status.replace('closed_','') }}</span></td>
            </tr>
            {% endfor %}
            </tbody>
          </table>
          {% else %}
          <p class="p-3 mb-0" style="color:#cdd5df">No closed trades yet.</p>
          {% endif %}
        </div>
      </div>
    </div>

    <!-- Cron log -->
    <div class="col-12">
      <div class="card">
        <div class="card-header">Run Log (last 80 lines)</div>
        <div class="card-body p-2">
          <pre id="log">{{ log_lines | join('\n') }}</pre>
        </div>
      </div>
    </div>

  </div>
</div>
<script>
  // Scroll log to bottom on load
  const el = document.getElementById('log');
  el.scrollTop = el.scrollHeight;
</script>
</body>
</html>
"""

def normalize(trade):
    return {
        "ticker":      trade.get("ticker", "?"),
        "direction":   trade.get("direction", "long"),
        "status":      trade.get("status", "open"),
        "entry_price": trade.get("entry_price") or 0.0,
        "exit_price":  trade.get("exit_price")  or 0.0,
        "entry_date":  trade.get("entry_date", ""),
        "exit_date":   trade.get("exit_date", ""),
        "score":       trade.get("score") or 0.0,
        "signals":     trade.get("signals") or [],
        "pnl_pct":     trade.get("pnl_pct") or 0.0,
        "pnl_usd":     trade.get("pnl_usd") or 0.0,
    }

@app.route("/")
def index():
    trades = [normalize(t) for t in load_json(TRADES_FILE)]
    queue  = [normalize(q) for q in load_json(QUEUE_FILE)]

    open_trades   = [t for t in trades if t["status"] == "open"]
    closed_trades = [t for t in trades if t["status"] != "open"]
    closed_trades.sort(key=lambda t: t["exit_date"], reverse=True)
    closed_trades = closed_trades[:20]

    total_pnl = sum(t["pnl_usd"] for t in trades if t["status"] != "open")
    equity    = TOTAL_CAPITAL + total_pnl

    return render_template_string(
        TEMPLATE,
        open_trades   = open_trades,
        closed_trades = closed_trades,
        queue         = queue,
        open_count    = len(open_trades),
        closed_count  = sum(1 for t in trades if t["status"] != "open"),
        total_pnl     = total_pnl,
        equity        = equity,
        total_capital = TOTAL_CAPITAL,
        log_lines     = load_log(),
        now           = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC"),
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)