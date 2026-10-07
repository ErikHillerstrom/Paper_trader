"""Long-running service: scheduler (daily report + alert scan) and the password-protected dashboard."""
from __future__ import annotations

import hmac
import json
import logging
import os
import secrets
import threading
from datetime import datetime
from zoneinfo import ZoneInfo

import yaml
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from flask import (Flask, Response, abort, flash, redirect, render_template, request, session, url_for)

from . import alerts, config, daily, db

log = logging.getLogger(__name__)
_run_lock = threading.Lock()
scheduler: BackgroundScheduler | None = None


# ---- jobs ------------------------------------------------------------------
def run_job(name: str, send: bool = True) -> str:
    """Run 'daily' or 'alerts' once, never two at the same time. Records result in the runs table."""
    if not _run_lock.acquire(blocking=False):
        return "another job is already running"
    rid = db.start_run(name)
    try:
        cfg = config.load_config()
        if name == "daily":
            r = daily.run(cfg, send=send)
            msg = f"{r['signal']['block'].splitlines()[0]} | email: {r.get('email_status', 'skipped')}"
        else:
            r = alerts.run(cfg, send=send)
            msg = f"{len(r['new'])} new alert(s); {r['checked']} condition(s) active; issues: {len(r['issues'])}"
        db.finish_run(rid, "ok", msg)
        return msg
    except Exception as e:  # noqa: BLE001
        log.exception("job %s failed", name)
        db.finish_run(rid, "failed", repr(e))
        return f"failed: {e!r}"
    finally:
        _run_lock.release()


def apply_schedule(cfg: dict) -> None:
    tz = ZoneInfo(cfg["timezone"])
    d, a = cfg["daily"], cfg["alerts"]
    h, m = d["time"].split(":")
    scheduler.add_job(run_job, CronTrigger(day_of_week=config.days_for_cron(d["days"]), hour=int(h), minute=int(m),
                                           timezone=tz), args=["daily"], id="daily", replace_existing=True,
                      coalesce=True, misfire_grace_time=3600, max_instances=1)
    if a["enabled"]:
        scheduler.add_job(run_job, CronTrigger(day_of_week=config.days_for_cron(a["days"]),
                                               hour=f"{int(a['start_hour'])}-{int(a['end_hour'])}/{int(a['every_hours'])}",
                                               minute=int(a["minute"]), timezone=tz), args=["alerts"], id="alerts",
                          replace_existing=True, coalesce=True, misfire_grace_time=1800, max_instances=1)
    elif scheduler.get_job("alerts"):
        scheduler.remove_job("alerts")


def catch_up(cfg: dict) -> None:
    """After a reboot: if today's report is missing and the scheduled time has passed, run it once."""
    tz = ZoneInfo(cfg["timezone"])
    now = datetime.now(tz)
    h, m = map(int, cfg["daily"]["time"].split(":"))
    if (cfg["daily"]["catch_up"] and now.weekday() in config.parse_days(cfg["daily"]["days"])
            and (now.hour, now.minute) > (h, m) and now.hour < 20 and not db.get_report(now.strftime("%Y-%m-%d"))):
        log.info("catch-up: today's daily report is missing, running it now")
        threading.Timer(20, run_job, args=["daily"]).start()


# ---- web app -----------------------------------------------------------------
def create_app() -> Flask:
    app = Flask(__name__, template_folder=str(config.ROOT / "monitor" / "templates"))
    key_file = config.DATA_DIR / "secret.key"
    if not key_file.exists():
        config.DATA_DIR.mkdir(parents=True, exist_ok=True)
        key_file.write_text(secrets.token_hex(32))
        key_file.chmod(0o600)
    app.secret_key = key_file.read_text().strip()
    app.config.update(SESSION_COOKIE_HTTPONLY=True, SESSION_COOKIE_SAMESITE="Strict")

    @app.before_request
    def guard():
        cfg = config.load_config()
        pw = os.environ.get("WEB_PASSWORD", "")
        auth = request.authorization
        ok = (auth and pw and hmac.compare_digest(auth.username or "", cfg["web"]["username"])
              and hmac.compare_digest(auth.password or "", pw))
        if not ok:
            return Response("Login required", 401, {"WWW-Authenticate": 'Basic realm="Nasdaq monitor"'})
        if request.method == "POST":
            if not hmac.compare_digest(request.form.get("csrf", ""), session.get("csrf", "x")):
                abort(400, "bad CSRF token - reload the page and try again")

    @app.after_request
    def headers(resp):
        resp.headers["X-Frame-Options"] = "DENY"
        resp.headers["X-Content-Type-Options"] = "nosniff"
        resp.headers["Cache-Control"] = "no-store"
        return resp

    @app.context_processor
    def inject():
        if "csrf" not in session:
            session["csrf"] = secrets.token_hex(16)
        return {"csrf": session["csrf"]}

    @app.get("/")
    def dashboard():
        rep = db.get_report()
        data = json.loads(rep["data_json"]) if rep else None
        jobs = [(j.id, j.next_run_time) for j in scheduler.get_jobs()] if scheduler else []
        return render_template("dashboard.html", rep=rep, data=data, alerts=db.list_alerts(8), runs=db.list_runs(6),
                               jobs=jobs, today=datetime.now().strftime("%Y-%m-%d"))

    @app.get("/reports")
    def reports():
        return render_template("reports.html", rows=db.list_reports())

    @app.get("/report/<date>")
    def report(date):
        rep = db.get_report(date)
        if not rep:
            abort(404)
        return render_template("report.html", rep=rep)

    @app.get("/alerts")
    def alerts_page():
        return render_template("alerts.html", rows=db.list_alerts(200))

    @app.get("/data")
    def data_page():
        sym = config.load_config()["symbol"]
        px, vx, fg = db.load_prices(sym), db.load_prices("VIX"), db.load_fng()
        return render_template("data.html", sym=sym, px=px.tail(15)[::-1].items(), vx=vx.tail(15)[::-1].items(),
                               fg=fg.tail(15)[::-1].items(), counts=(len(px), len(vx), len(fg)))

    @app.get("/logs")
    def logs():
        p = config.DATA_DIR / "monitor.log"
        text = "\n".join(p.read_text(errors="replace").splitlines()[-200:]) if p.exists() else "(no log yet)"
        return render_template("logs.html", text=text, runs=db.list_runs(30))

    @app.post("/run/<job>")
    def run_now(job):
        if job not in ("daily", "alerts"):
            abort(404)
        send = request.form.get("send") == "1"
        threading.Thread(target=run_job, args=(job, send), daemon=True).start()
        flash(f"Started '{job}' job{' (email on)' if send else ' (no email)'}. Refresh in a minute to see the result.")
        return redirect(url_for("dashboard"))

    @app.route("/settings", methods=["GET", "POST"])
    def settings():
        cfg = config.load_config()
        if request.method == "POST":
            form = request.form
            if form.get("mode") == "raw":
                try:
                    new = config.deep_merge(config.DEFAULTS, yaml.safe_load(form["raw"]) or {})
                except yaml.YAMLError as e:
                    flash(f"YAML error: {e}")
                    return render_template("settings.html", cfg=cfg, raw=form["raw"])
            else:
                new = config.load_config()
                try:
                    new["daily"]["time"] = form["daily_time"].strip()
                    new["daily"]["days"] = form["daily_days"].strip()
                    new["daily"]["send_email"] = form.get("daily_send") == "on"
                    new["daily"]["catch_up"] = form.get("daily_catchup") == "on"
                    a = new["alerts"]
                    a["enabled"] = form.get("alerts_enabled") == "on"
                    a["days"] = form["alerts_days"].strip()
                    for k in ("every_hours", "start_hour", "end_hour", "minute"):
                        a[k] = int(form[f"alerts_{k}"])
                    for k in a["thresholds"]:
                        a["thresholds"][k] = float(form[f"th_{k}"])
                    new["email"]["to"] = [x.strip() for x in form["email_to"].replace(";", ",").split(",") if x.strip()]
                    new["email"]["enabled"] = form.get("email_enabled") == "on"
                except (KeyError, ValueError) as e:
                    flash(f"Invalid form value: {e}")
                    return redirect(url_for("settings"))
            errs = config.save_config(new)
            if errs:
                flash("Not saved: " + "; ".join(errs))
                return render_template("settings.html", cfg=cfg, raw=form.get("raw") or yaml.safe_dump(cfg, sort_keys=False))
            apply_schedule(new)
            flash("Saved. The schedule has been updated.")
            return redirect(url_for("settings"))
        return render_template("settings.html", cfg=cfg, raw=yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))

    return app


def main() -> None:
    global scheduler
    config.load_env()
    cfg = config.load_config()
    if not os.environ.get("WEB_PASSWORD"):
        raise SystemExit("Set WEB_PASSWORD in .env before starting the web service (it refuses to run without one).")
    db.init()
    scheduler = BackgroundScheduler(timezone=ZoneInfo(cfg["timezone"]))
    apply_schedule(cfg)
    scheduler.start()
    catch_up(cfg)
    for j in scheduler.get_jobs():
        log.info("scheduled %s, next run %s", j.id, j.next_run_time)
    app = create_app()
    host, port = cfg["web"]["host"], int(cfg["web"]["port"])
    log.info("dashboard on http://%s:%s", host, port)
    try:
        from waitress import serve
        serve(app, host=host, port=port, threads=4)
    except ImportError:
        app.run(host=host, port=port)
