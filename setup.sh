#!/usr/bin/env bash
# One-time setup on Linux Mint / Ubuntu / Debian. Run from the project folder: ./setup.sh
set -euo pipefail
cd "$(dirname "$0")"
DIR="$(pwd)"

command -v python3 >/dev/null || { echo "python3 missing: sudo apt install python3"; exit 1; }
python3 -c 'import sys; assert sys.version_info >= (3,9), "Python 3.9+ needed"' || exit 1
python3 -c 'import venv, ensurepip' 2>/dev/null || { echo "Run: sudo apt install python3-venv python3-pip"; exit 1; }

[ -d .venv ] || python3 -m venv .venv
.venv/bin/pip install --quiet --upgrade pip
.venv/bin/pip install --quiet -r requirements.txt
echo "Dependencies installed."

if [ ! -f .env ]; then
  cat > .env <<ENV
# Gmail: create an App Password (Google Account > Security > 2-Step Verification > App passwords)
EMAIL_PASSWORD=
# Password for the web dashboard (username is web.username in config.yaml, default 'admin')
WEB_PASSWORD=$(python3 -c 'import secrets; print(secrets.token_urlsafe(12))')
ENV
  chmod 600 .env
  echo "Created .env (edit it: add EMAIL_PASSWORD; a random WEB_PASSWORD was generated)."
fi

sed -e "s|__USER__|$(id -un)|g" -e "s|__DIR__|$DIR|g" deploy/nasdaq-monitor.service > deploy/nasdaq-monitor.service.generated
.venv/bin/python -m unittest discover -s tests 2>&1 | tail -3

cat <<MSG

Next steps:
  1. Edit config.yaml (email address, times) and .env (EMAIL_PASSWORD).
  2. Check the data sources from this machine:   .venv/bin/python -m monitor check-sources
  3. Try the email:                              .venv/bin/python -m monitor test-email
  4. Try a full report without emailing:         .venv/bin/python -m monitor daily --no-email
  5. Install as an always-on service:
       sudo cp deploy/nasdaq-monitor.service.generated /etc/systemd/system/nasdaq-monitor.service
       sudo systemctl daemon-reload && sudo systemctl enable --now nasdaq-monitor
       journalctl -u nasdaq-monitor -f        # watch it start
  6. Open the dashboard: http://<laptop-ip>:8080   (see web.port in config.yaml)
MSG
