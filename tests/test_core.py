"""Invariant tests.

Run with ``pytest -q`` or directly: ``python tests/test_core.py``.

These assert properties that must hold for the analysis to mean anything. They
are deliberately about *correctness of reasoning*, not just "the code runs":
if the arithmetic identity breaks, or precision and recall stop trading off,
or peeking stops inflating error, then something in the study is wrong.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from experiments import (
    cuped_adjust,
    required_n_per_arm,
    simulate_cuped,
    simulate_peeking,
    srm_check,
)
from leakage import leakage_demo
from synth import generate, walk_forward_windows
from two_stage import evaluate_window, threshold_sweep

SMALL = 300_000


def test_arithmetic_identity():
    """ltv7 must equal ltv2 + delta exactly. The leakage demo depends on it."""
    df = generate(n=50_000, seed=1)
    assert np.allclose(df.ltv7, df.ltv2 + df.delta_ltv7, atol=1e-6)


def test_marginals_are_realistic():
    df = generate(n=SMALL, seed=1)
    payer_rate = (df.ltv2 > 0).mean()
    repeat_rate = (df.loc[df.ltv2 > 0, "delta_ltv7"] > 0).mean()
    assert 0.003 < payer_rate < 0.008, payer_rate
    assert 0.40 < repeat_rate < 0.60, repeat_rate
    # The tail must actually be heavy, or calibration has nothing to struggle with.
    whale_share = df.loc[df.is_whale, "delta_ltv7"].sum() / df.delta_ltv7.sum()
    assert whale_share > 0.15, whale_share


def test_walk_forward_has_no_temporal_overlap():
    df = generate(n=SMALL, seed=1)
    windows = walk_forward_windows(df)
    assert len(windows) >= 3
    for train, test in windows:
        assert train.install_date.max() < test.install_date.min()


def test_embargo_respects_label_horizon():
    """No training install may sit within the label horizon of the test start."""
    df = generate(n=SMALL, seed=1)
    for train, test in walk_forward_windows(df, embargo_days=7):
        gap = (test.install_date.min() - train.install_date.max()).days
        assert gap >= 7, gap


def test_leaked_model_is_beaten_by_copying_a_column():
    """The punchline: the leak's apparent skill is not skill."""
    res = leakage_demo(generate(n=SMALL, seed=1))
    allu = res[res.population == "all users"].set_index("setup")
    copy = allu.loc["copy ltv2 forward (no model)", "MAE"]
    leaked = allu.loc["ltv2 IN features -> ltv7  [LEAKED]", "MAE"]
    honest = allu.loc["ltv2 dropped     -> ltv7  [partial]", "MAE"]
    assert copy < leaked, (copy, leaked)      # no model beats the leaked model
    assert leaked < honest, (leaked, honest)  # and the leak still flatters it


def test_precision_and_recall_trade_off():
    """The invariant that makes 'high precision AND high recall' impossible."""
    rng = np.random.default_rng(0)
    label = rng.binomial(1, 0.5, 20_000)
    proba = np.clip(0.5 + 0.25 * (2 * label - 1) + rng.normal(0, 0.25, 20_000), 0, 1)
    s = threshold_sweep(proba, label, grid=np.linspace(0.1, 0.9, 17))
    assert s.precision.is_monotonic_increasing, s
    assert s.recall.is_monotonic_decreasing, s


def test_calibration_reduces_bias_on_average():
    """Isotonic must cut aggregate bias across seeds, not just on a lucky one."""
    uncal, cal = [], []
    for seed in (1, 2, 3):
        df = generate(n=SMALL, seed=seed)
        for train, test in walk_forward_windows(df):
            r = evaluate_window(train, test)
            uncal.append(abs(r["total_error_uncal"]))
            cal.append(abs(r["total_error_cal"]))
    assert np.mean(cal) < np.mean(uncal), (np.mean(cal), np.mean(uncal))


def test_cuped_is_unbiased_and_reduces_variance():
    res = simulate_cuped(n=60_000, true_lift=0.02, rho=0.7, seed=3).set_index("method")
    # Both recover the true +2.0 effect; CUPED does it with a tighter interval.
    assert abs(res.loc["raw", "diff"] - 2.0) < 0.5
    assert abs(res.loc["cuped", "diff"] - 2.0) < 0.5
    assert res.loc["cuped", "se"] < res.loc["raw", "se"]
    # Variance reduction should land near rho^2 = 0.49.
    assert 0.40 < res.loc["cuped", "variance_reduction"] < 0.58


def test_cuped_theta_matches_closed_form():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 10_000)
    y = 3.0 * x + rng.normal(0, 1, 10_000)
    _, theta, reduction = cuped_adjust(y, x)
    assert abs(theta - 3.0) < 0.1, theta
    assert 0.85 < reduction < 0.95, reduction


def test_srm_detects_a_broken_split_and_passes_a_clean_one():
    assert not srm_check(50_012, 49_988)["srm_detected"]
    assert srm_check(51_500, 48_500)["srm_detected"]


def test_power_scales_the_right_way():
    """Smaller effects need more users; the relationship is roughly 1/MDE^2."""
    big = required_n_per_arm(0.05, 0.10)
    small = required_n_per_arm(0.05, 0.05)
    assert small > big
    assert 3.0 < small / big < 5.0, small / big


def test_peeking_inflates_false_positives():
    res = simulate_peeking(n_sims=400, n_peeks=10, seed=5).set_index("method")
    fixed = res.loc["fixed-horizon, 10 peeks", "false_positive_rate"]
    seq = res.loc["always-valid confidence sequence", "false_positive_rate"]
    assert fixed > 0.10, fixed   # well above the nominal 5%
    assert seq <= 0.05, seq      # stays valid under continuous monitoring


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL  {fn.__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
