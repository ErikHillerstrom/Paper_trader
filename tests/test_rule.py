import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from monitor import rule  # noqa: E402


def make_prices(kind: str, n: int = 420) -> pd.Series:
    idx = pd.bdate_range("2024-01-01", periods=n)
    x = np.arange(n)
    if kind == "up":
        v = 100 + x * 0.3
    elif kind == "down_after_up":       # rise for 250 days, then fall hard -> death cross
        v = np.where(x < 250, 100 + x * 0.3, 175 - (x - 250) * 0.9)
    elif kind == "flat_then_golden":    # long fall, then sharp recovery -> golden cross
        v = np.where(x < 250, 200 - x * 0.3, 125 + (x - 250) * 1.2)
    else:
        raise ValueError(kind)
    return pd.Series(v, index=idx)


def fng_series(closes, default=40.0, overrides=None):
    s = pd.Series(default, index=closes.index, dtype=float)
    for k, v in (overrides or {}).items():
        s.loc[k] = v
    return s


class RuleTests(unittest.TestCase):
    def death_cross_date(self, closes):
        # evaluate with an unlimited fear provider just to find D
        r = rule.evaluate(closes, rule.FngProvider(fng_series(closes)))
        return pd.Timestamp(r["death_cross_date"])

    def test_uptrend_invested(self):
        c = make_prices("up")
        r = rule.evaluate(c, rule.FngProvider(fng_series(c)))
        self.assertEqual(r["target"], "INVESTED")
        self.assertEqual(r["branch"], "A")

    def test_downtrend_normal_exit(self):
        c = make_prices("down_after_up")
        r = rule.evaluate(c, rule.FngProvider(fng_series(c, 40)))
        self.assertEqual(r["trend"], "downtrend")
        self.assertEqual(r["target"], "CASH")
        self.assertEqual(r["branch"], "B2")

    def test_boundary_25_is_normal_exit(self):
        c = make_prices("down_after_up")
        D = self.death_cross_date(c)
        pos = c.index.get_loc(D)
        s = fng_series(c, 40, {c.index[pos - 3]: 25.0})
        r = rule.evaluate(c, rule.FngProvider(s))
        self.assertEqual((r["target"], r["branch"]), ("CASH", "B2"))

    def test_skipped_cross_stays_invested(self):
        c = make_prices("down_after_up")
        D = self.death_cross_date(c)
        pos = c.index.get_loc(D)
        s = fng_series(c, 40, {c.index[pos - 3]: 24.9})
        r = rule.evaluate(c, rule.FngProvider(s))
        self.assertEqual((r["target"], r["branch"]), ("INVESTED", "B3-skipped"))
        self.assertTrue(any("points below the 50" in w for w in r["watch"]))

    def test_skipped_cross_exits_when_fear_fades(self):
        c = make_prices("down_after_up")
        D = self.death_cross_date(c)
        pos = c.index.get_loc(D)
        s = fng_series(c, 40, {c.index[pos - 3]: 10.0, c.index[pos + 5]: 50.0})
        r = rule.evaluate(c, rule.FngProvider(s))
        self.assertEqual((r["target"], r["branch"]), ("CASH", "B3-faded"))
        # but evaluated as of the day BEFORE the fade, still invested
        r2 = rule.evaluate(c, rule.FngProvider(s), as_of=c.index[pos + 4])
        self.assertEqual(r2["target"], "INVESTED")

    def test_missing_fear_data_is_uncertain(self):
        c = make_prices("down_after_up")
        s = fng_series(c, 40).iloc[:-120]    # history stops long before the cross
        r = rule.evaluate(c, rule.FngProvider(s))
        self.assertEqual(r["target"], "UNCERTAIN")
        self.assertEqual(r["reason_code"], "fear_data")

    def test_gap_in_post_cross_data_is_uncertain_not_guessed(self):
        c = make_prices("down_after_up")
        D = self.death_cross_date(c)
        pos = c.index.get_loc(D)
        s = fng_series(c, 40, {c.index[pos - 3]: 10.0}).drop(c.index[pos + 4])
        r = rule.evaluate(c, rule.FngProvider(s))
        self.assertEqual(r["target"], "UNCERTAIN")

    def test_too_little_history(self):
        c = make_prices("up", 150)
        r = rule.evaluate(c, rule.FngProvider(fng_series(c)))
        self.assertEqual(r["target"], "UNCERTAIN")

    def test_golden_cross_back_to_invested(self):
        c = make_prices("flat_then_golden")
        r = rule.evaluate(c, rule.FngProvider(fng_series(c)))
        self.assertEqual(r["target"], "INVESTED")
        self.assertEqual(r["last_cross"]["type"], "golden")

    def test_action_matrix(self):
        I, C, U = {"target": "INVESTED"}, {"target": "CASH"}, {"target": "UNCERTAIN"}
        self.assertEqual(rule.decide_action(C, I), "BUY at tomorrow's open")
        self.assertEqual(rule.decide_action(I, C), "SELL at tomorrow's open")
        self.assertEqual(rule.decide_action(I, I), "HOLD")
        self.assertEqual(rule.decide_action(C, C), "STAY IN CASH")
        self.assertEqual(rule.decide_action(U, I), "UNCERTAIN - no action")

    def test_death_cross_watch_in_uptrend(self):
        # slow decline keeping SMA50 just above SMA200
        n = 420
        idx = pd.bdate_range("2024-01-01", periods=n)
        x = np.arange(n)
        v = np.where(x < 330, 100 + x * 0.3, 199 - (x - 330) * 0.35)
        c = pd.Series(v, index=idx)
        r = rule.evaluate(c, rule.FngProvider(fng_series(c)))
        if r["trend"] == "uptrend" and r["gap_pct"] < 2:
            self.assertTrue(any("Death cross may be approaching" in w for w in r["watch"]))

    def test_vix_fallback_runs(self):
        c = make_prices("down_after_up", 520)
        v = pd.Series(15.0, index=c.index)
        r = rule.evaluate(c, rule.VixProvider(v))
        # flat VIX -> never in top quartile at cross -> normal exit
        self.assertEqual((r["target"], r["branch"]), ("CASH", "B2"))


if __name__ == "__main__":
    unittest.main()
