# Nasdaq-100 monitor

Two jobs plus a local dashboard, all in one always-on service (`python -m monitor serve`):

- **Daily report** (weekdays 08:30 Stockholm by default): downloads QQQ history (Yahoo, Stooq as fallback), VIX (FRED),
  CNN Fear & Greed daily history and news (RSS), stores everything in `data/monitor.db`, computes your signal with the
  exact rule (`monitor/rule.py`), saves the report and emails it.
- **Alert scan** (every 3 hours, 09:05-21:05 by default): reads fresh prices from Yahoo and Fear & Greed from the
  database, emails only when something crosses a threshold (QQQ move, VIX spike/level, close near SMA50/200,
  Fear & Greed jump or extreme, big member move, BUY/SELL signal). Each alert is sent once.
- **Dashboard** (`http://<laptop-ip>:8080`, password protected): today's report, history, alerts, stored data, logs,
  and a Settings page for the schedule, thresholds and recipients (changes apply immediately, no restart).

## Install (Linux Mint)
    sudo apt install python3-venv python3-pip
    ./setup.sh
Then follow the printed steps. Put `EMAIL_PASSWORD` (Gmail app password) and `WEB_PASSWORD` in `.env`.

## Useful commands
    .venv/bin/python -m monitor check-sources    # tests every data source from this machine
    .venv/bin/python -m monitor test-email
    .venv/bin/python -m monitor daily --no-email
    .venv/bin/python -m monitor alerts --no-email
    .venv/bin/python -m unittest discover -s tests   # rule + end-to-end tests (offline)

## Things to know
- The signal uses only the rule. If a number is missing it prints `SIGNAL: UNCERTAIN - no action`. If Fear & Greed
  history is incomplete it falls back to the VIX version of the rule and says so.
- Uses unadjusted QQQ closes. The sample report calls the previous completed session "today's close".
- Free data sources are unofficial (Yahoo, Stooq, CNN's feed) and can change; run `check-sources` if a report looks wrong.
  The daily email lists data issues, and the alert scan warns if stored Fear & Greed data goes stale.
- Without `EMAIL_PASSWORD`, emails are saved to `data/outbox/` instead of sent (dry run).
- The dashboard uses HTTP basic auth over plain HTTP: keep it on your home network, or reach it via a VPN such as
  Tailscale. Never port-forward it to the internet. Set `web.host: 127.0.0.1` to limit it to the laptop itself.
- Delete `data/` to reset history; `config.yaml.bak` is the previous config after a save from the web UI.
- Not financial advice: the signal is the mechanical output of your own rule.
