"""Online experiment utilities: SRM, power, CUPED, and sequential testing.

Four things that decide whether an A/B result means anything:

* **SRM** -- if the realised split does not match the intended one, assignment
  is broken and every downstream number is suspect. Check before reading results.
* **Power** -- an underpowered test that comes back flat has told you nothing,
  but is routinely reported as "no effect".
* **CUPED** -- pre-experiment data explains part of the outcome variance.
  Removing it shrinks confidence intervals at no cost in bias, which is the
  cheapest sensitivity gain available.
* **Sequential testing** -- repeatedly checking a fixed-horizon test inflates
  the false positive rate badly. Either stop peeking or use a bound that
  remains valid under continuous monitoring.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


# --------------------------------------------------------------------------
# Sample ratio mismatch
# --------------------------------------------------------------------------

def srm_check(n_control: int, n_treatment: int, expected_ratio: float = 0.5) -> dict:
    """Chi-square test that the realised split matches the intended one.

    A failing SRM invalidates the experiment. It usually means broken
    assignment, a redirect dropping users, or differential bot filtering --
    not an effect worth interpreting.
    """
    total = n_control + n_treatment
    expected = np.array([total * (1 - expected_ratio), total * expected_ratio])
    observed = np.array([n_control, n_treatment])
    chi2 = float(((observed - expected) ** 2 / expected).sum())
    p = float(stats.chi2.sf(chi2, df=1))
    return {
        "n_control": n_control,
        "n_treatment": n_treatment,
        "observed_ratio": n_treatment / total,
        "chi2": chi2,
        "p_value": p,
        "srm_detected": p < 0.001,  # conventional threshold; SRM is a red flag, not a finding
    }


# --------------------------------------------------------------------------
# Power
# --------------------------------------------------------------------------

def required_n_per_arm(
    baseline_rate: float, mde_relative: float, alpha: float = 0.05, power: float = 0.8
) -> int:
    """Users per arm needed to detect a relative lift on a binary metric."""
    p1 = baseline_rate
    p2 = baseline_rate * (1 + mde_relative)
    pbar = (p1 + p2) / 2
    z_a = stats.norm.ppf(1 - alpha / 2)
    z_b = stats.norm.ppf(power)
    num = (z_a * np.sqrt(2 * pbar * (1 - pbar)) + z_b * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2
    return int(np.ceil(num / (p2 - p1) ** 2))


# --------------------------------------------------------------------------
# CUPED
# --------------------------------------------------------------------------

def cuped_adjust(y: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Variance reduction using a pre-experiment covariate.

    ``y_adj = y - theta * (x - mean(x))`` with ``theta = cov(y, x) / var(x)``.

    The adjustment is unbiased because ``x`` is measured before assignment and
    so cannot be affected by the treatment; subtracting a mean-zero quantity
    leaves the expected difference between arms unchanged while removing the
    part of the variance ``x`` explains. Variance falls by roughly ``rho^2``,
    so a pre-period covariate correlated 0.7 with the outcome removes about
    half the variance -- equivalent to doubling the sample at no cost.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    theta = float(np.cov(y, x, ddof=1)[0, 1] / np.var(x, ddof=1))
    y_adj = y - theta * (x - x.mean())
    reduction = 1 - float(np.var(y_adj, ddof=1) / np.var(y, ddof=1))
    return y_adj, theta, reduction


def welch_test(a: np.ndarray, b: np.ndarray) -> dict:
    """Two-sided Welch t-test, returning the effect and its interval."""
    t, p = stats.ttest_ind(b, a, equal_var=False)
    diff = float(np.mean(b) - np.mean(a))
    se = float(np.sqrt(np.var(a, ddof=1) / len(a) + np.var(b, ddof=1) / len(b)))
    return {
        "diff": diff,
        "relative": diff / float(np.mean(a)) if np.mean(a) else float("nan"),
        "se": se,
        "ci_low": diff - 1.96 * se,
        "ci_high": diff + 1.96 * se,
        "p_value": float(p),
        "significant": bool(p < 0.05),
    }


def simulate_cuped(
    n: int = 40_000, true_lift: float = 0.02, rho: float = 0.7, seed: int = 0
) -> pd.DataFrame:
    """Same experiment analysed with and without CUPED."""
    rng = np.random.default_rng(seed)
    pre = rng.normal(100, 20, n)
    noise = rng.normal(0, 20 * np.sqrt(1 - rho**2), n)
    post = rho * (pre - 100) + 100 + noise
    arm = rng.random(n) < 0.5
    post = post + arm * true_lift * 100  # treatment effect applied post-assignment only

    adj, _, reduction = cuped_adjust(post, pre)
    raw = welch_test(post[~arm], post[arm])
    cup = welch_test(adj[~arm], adj[arm])
    return pd.DataFrame(
        [
            {"method": "raw", **raw, "variance_reduction": 0.0},
            {"method": "cuped", **cup, "variance_reduction": reduction},
        ]
    )


# --------------------------------------------------------------------------
# Sequential testing
# --------------------------------------------------------------------------

def confidence_sequence_halfwidth(
    t: np.ndarray, sigma: float, alpha: float = 0.05, rho: float = 1.0
) -> np.ndarray:
    """Normal-mixture confidence sequence half-width (Howard et al.).

    A fixed-horizon interval is valid at one pre-committed sample size. A
    confidence sequence is valid at *every* sample size simultaneously, so it
    can be monitored continuously without inflating the error rate. The price
    is a wider interval: it must hold uniformly over time, not just once.
    """
    t = np.asarray(t, dtype=float)
    inner = (t * rho**2 + 1) / (t**2 * rho**2)
    return sigma * np.sqrt(2 * inner * np.log(np.sqrt(t * rho**2 + 1) / alpha))


def simulate_peeking(
    n_sims: int = 2000, n_max: int = 10_000, n_peeks: int = 10, alpha: float = 0.05, seed: int = 0
) -> pd.DataFrame:
    """False positive rate under A/A, fixed-horizon vs always-valid.

    Both arms are drawn from the same distribution, so every "significant"
    result is a false positive. The fixed-horizon test is applied at each of
    ``n_peeks`` interim looks and called significant if it ever crosses -- the
    standard way teams accidentally run experiments.
    """
    rng = np.random.default_rng(seed)
    checkpoints = np.linspace(n_max // n_peeks, n_max, n_peeks).astype(int)
    z_crit = stats.norm.ppf(1 - alpha / 2)

    fixed_hits = 0
    seq_hits = 0
    for _ in range(n_sims):
        a = rng.normal(0, 1, n_max)
        b = rng.normal(0, 1, n_max)
        ca = np.cumsum(a)
        cb = np.cumsum(b)
        fixed_flag = False
        seq_flag = False
        for t in checkpoints:
            diff = cb[t - 1] / t - ca[t - 1] / t
            se = np.sqrt(2.0 / t)
            if abs(diff / se) > z_crit:
                fixed_flag = True
            hw = confidence_sequence_halfwidth(np.array([t]), sigma=np.sqrt(2.0), alpha=alpha)[0]
            if abs(diff) > hw:
                seq_flag = True
        fixed_hits += fixed_flag
        seq_hits += seq_flag

    return pd.DataFrame(
        [
            {
                "method": f"fixed-horizon, {n_peeks} peeks",
                "false_positive_rate": fixed_hits / n_sims,
                "nominal_alpha": alpha,
            },
            {
                "method": "always-valid confidence sequence",
                "false_positive_rate": seq_hits / n_sims,
                "nominal_alpha": alpha,
            },
        ]
    )
