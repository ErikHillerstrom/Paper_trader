"""Offline end-to-end test: fake data sources, real DB / report / alerts / web UI."""
import os, sys, tempfile, unittest
from datetime import datetime, timezone
from pathlib import Path
import numpy as np, pandas as pd

TMP = tempfile.mkdtemp()
os.environ["MONITOR_DATA"] = TMP
os.environ["WEB_PASSWORD"] = "pw"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from monitor import alerts, config, daily, db, service, sources  # noqa: E402

N = 520
IDX = pd.bdate_range(end=pd.Timestamp.now().normalize() - pd.Timedelta(days=1), periods=N)
X = np.arange(N)


def fake_daily(symbol, period="3y"):
    if symbol == "^VIX":
        return pd.Series(15.0 + (X % 7), index=IDX), "fake"
    v = pd.Series(np.where(X < 380, 100 + X * 0.3, 214 - (X - 380) * 0.9), index=IDX)  # death cross late
    if period == "1y":   # alert scan: make last day a -3% move
        v.iloc[-1] = v.iloc[-2] * 0.97
    return v, "fake"


def fake_fng(start=None):
    return {d.strftime("%Y-%m-%d"): (40.0 if i != 400 else 12.0, "fear") for i, d in enumerate(IDX)}, None


class Flow(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sources.fetch_daily = fake_daily
        sources.fetch_vix = lambda: (fake_daily("^VIX")[0], "fake")
        sources.fetch_fng = fake_fng
        sources.fetch_news = lambda feeds, kws, hrs, now=None: ([{
            "title": "Nvidia beats estimates", "link": "http://x/1", "source": "T",
            "published": datetime.now(timezone.utc), "hits": 1}], [])
        sources.fetch_earnings_today = lambda t, d: ["NVDA reports earnings tomorrow"]
        alerts._members_moves = lambda m: {"NVDA": 6.2, "AAPL": 0.1}
        db.init()
        cls.cfg = config.deep_merge(config.load_config(), {"email": {"enabled": False}})

    def test_1_daily_report(self):
        r = daily.run(self.cfg, send=True)
        text = daily.render_text(r)
        self.assertIn("SIGNAL:", text)
        self.assertIn("Target position:", text)
        self.assertTrue(db.get_report(r["date"]))
        print("\n" + text[:1500])
        self.assertTrue(list((Path(TMP) / "outbox").glob("*.txt")))  # dry-run email saved

    def test_2_alerts_dedupe(self):
        res = alerts.run(self.cfg, send=True)
        self.assertTrue(any("QQQ" in s for s in res["new"]), res)
        self.assertTrue(any("NVDA" in s for s in res["new"]), res)
        res2 = alerts.run(self.cfg, send=True)
        self.assertEqual(res2["new"], [])   # same day -> no duplicates

    def test_3_web(self):
        service.scheduler = service.BackgroundScheduler()
        service.apply_schedule(self.cfg)
        service.scheduler.start(paused=True)
        app = service.create_app()
        c = app.test_client()
        self.assertEqual(c.get("/").status_code, 401)
        h = {"Authorization": "Basic " + __import__("base64").b64encode(b"admin:pw").decode()}
        for path in ["/", "/reports", "/alerts", "/data", "/logs", "/settings"]:
            self.assertEqual(c.get(path, headers=h).status_code, 200, path)
        self.assertEqual(c.post("/run/daily", headers=h, data={}).status_code, 400)   # CSRF enforced
        with c.session_transaction() as s:
            s["csrf"] = "tok"
        form = {"csrf": "tok", "mode": "form", "daily_time": "07:45", "daily_days": "mon-fri", "daily_send": "on",
                "alerts_enabled": "on", "alerts_days": "mon-fri", "alerts_every_hours": "2", "alerts_start_hour": "10",
                "alerts_end_hour": "22", "alerts_minute": "10", "email_to": "a@b.se", "email_enabled": "on"}
        for k, v in config.load_config()["alerts"]["thresholds"].items():
            form[f"th_{k}"] = str(v)
        r = c.post("/settings", headers=h, data=form)
        self.assertEqual(r.status_code, 302)
        self.assertEqual(service.scheduler.get_job("daily").trigger.fields[5].__str__(), "7")
        form["daily_time"] = "25:99"
        r = c.post("/settings", headers=h, data=form)
        self.assertIn(b"Not saved", r.data)
        service.scheduler.shutdown(wait=False)


if __name__ == "__main__":
    unittest.main()
