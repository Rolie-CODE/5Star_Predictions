import numpy as np
import pytest

from app.ml import backtest


def test_betting_record_at_minus_110():
    y = np.array([1, 1, 0, 0])
    p = np.array([0.7, 0.4, 0.3, 0.52])
    # Trust 0.70, 0.60, 0.70, 0.52 -> right, wrong, right, wrong.
    everything = backtest.betting_record(y, p, 0.5)
    assert everything["bets"] == 4 and everything["hit_rate"] == 0.5
    assert everything["roi"] == pytest.approx((2 * 100 / 110 - 2) / 4, abs=1e-4)
    confident = backtest.betting_record(y, p, 0.65)
    assert confident["bets"] == 2 and confident["roi"] == pytest.approx(100 / 110, abs=1e-4)
    assert backtest.betting_record(y, p, 0.99)["roi"] is None


def test_walk_forward_fits_only_on_earlier_seasons(monkeypatch):
    monkeypatch.setattr(backtest, "MIN_CALIBRATION_ROWS", 300)
    monkeypatch.setattr(backtest, "MIN_TEST_ROWS", 300)
    rng = np.random.default_rng(0)
    samples = [
        backtest.Sample(season, [float(x)], int(x > 0))
        for season in (2000, 2001, 2002, 2003)
        for x in rng.normal(size=300)
    ]
    seen = []
    real_fit = backtest.train.fit_calibrated

    def fake_fit(X_fit, y_fit, X_cal, y_cal):
        seen.append((len(y_fit), len(y_cal)))
        return real_fit(X_fit, y_fit, X_cal, y_cal)

    monkeypatch.setattr(backtest.train, "fit_calibrated", fake_fit)
    tested = [season for season, _, _ in backtest.walk_forward(samples, 2002)]
    assert tested == [2002, 2003]
    # 2002: fit 2000, calibrate 2001.  2003: fit 2000-2001, calibrate 2002.
    assert seen == [(300, 300), (600, 300)]
