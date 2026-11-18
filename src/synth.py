"""Synthetic zero-inflated LTV data for a free-to-play mobile game.

Generates D0-D2 in-game behaviour alongside early payment (``ltv2``) and the
D3-D7 payment increment (``delta_ltv7``). The point is to reproduce the three
properties that make real pLTV modelling hard:

1. Extreme zero inflation. Roughly 0.5% of installs pay at all.
2. A long right tail. A handful of "whales" dominate total revenue.
3. ``ltv7 = ltv2 + delta_ltv7`` by construction, so that putting ``ltv2`` in the
   feature set while predicting ``ltv7`` leaks the answer. See ``leakage.py``.

No real user data is used or reproduced anywhere in this repository.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import expit

# Target marginals, chosen to sit in the range typical of casual-game UA data.
P_EARLY_PAYER = 0.005          # share of installs paying during D0-D2
P_LATE_GIVEN_EARLY = 0.50      # share of early payers who pay again in D3-D7
P_LATE_GIVEN_NONE = 0.0012     # share of non-payers converting in D3-D7

FEATURES = [
    "sessions_3d",
    "playtime_min_3d",
    "levels_passed_3d",
    "levels_failed_3d",
    "items_used_3d",
    "revives_3d",
    "coin_net_3d",
    "iap_impressions_3d",
    "iap_clicks_3d",
    "fail_ratio",
    "revive_rate",
    "iap_ctr",
    "levels_per_session",
    "playtime_per_session",
]


def generate(
    n: int = 600_000,
    start: str = "2024-11-01",
    days: int = 180,
    seed: int = 7,
) -> pd.DataFrame:
    """Return one row per install with D0-D2 behaviour and payment outcomes."""
    rng = np.random.default_rng(seed)

    # Latent engagement. Everything observable is a noisy function of this, which
    # is what gives the behavioural features genuine (but imperfect) signal.
    z = rng.normal(0.0, 1.0, n)

    sessions = rng.poisson(np.exp(1.0 + 0.45 * z)) + 1
    playtime = rng.gamma(shape=2.0, scale=6.0 * np.exp(0.40 * z))
    levels_passed = rng.poisson(np.exp(1.1 + 0.55 * z))
    levels_failed = rng.poisson(np.exp(0.8 + 0.30 * z))
    items_used = rng.poisson(np.exp(0.2 + 0.50 * z))
    revives = rng.poisson(np.exp(-0.6 + 0.55 * z))
    coin_net = rng.normal(80.0 * z, 220.0)
    iap_impr = rng.poisson(np.exp(0.6 + 0.35 * z))
    iap_clicks = rng.binomial(iap_impr, expit(-2.0 + 0.45 * z))

    df = pd.DataFrame(
        {
            "install_date": pd.to_datetime(start)
            + pd.to_timedelta(rng.integers(0, days, n), unit="D"),
            "sessions_3d": sessions,
            "playtime_min_3d": playtime.round(2),
            "levels_passed_3d": levels_passed,
            "levels_failed_3d": levels_failed,
            "items_used_3d": items_used,
            "revives_3d": revives,
            "coin_net_3d": coin_net.round(1),
            "iap_impressions_3d": iap_impr,
            "iap_clicks_3d": iap_clicks,
        }
    )

    # Derived ratios. Cheap to compute, and they carry most of the lift in
    # practice because they normalise away raw volume.
    attempts = df.levels_passed_3d + df.levels_failed_3d
    df["fail_ratio"] = np.where(attempts > 0, df.levels_failed_3d / attempts, 0.0)
    df["revive_rate"] = np.where(attempts > 0, df.revives_3d / attempts, 0.0)
    df["iap_ctr"] = np.where(
        df.iap_impressions_3d > 0, df.iap_clicks_3d / df.iap_impressions_3d, 0.0
    )
    df["levels_per_session"] = df.levels_passed_3d / df.sessions_3d
    df["playtime_per_session"] = df.playtime_min_3d / df.sessions_3d

    # --- Early payment (D0-D2) -------------------------------------------------
    # Intercept solved numerically so the realised rate lands on P_EARLY_PAYER
    # regardless of how the covariate weights are retuned.
    pay_score = 1.25 * z + 2.0 * df.iap_ctr.to_numpy() + 0.25 * rng.normal(0, 1, n)
    intercept = _solve_intercept(pay_score, P_EARLY_PAYER)
    is_early = rng.random(n) < expit(intercept + pay_score)

    ltv2 = np.zeros(n)
    ltv2[is_early] = rng.lognormal(mean=1.6, sigma=1.1, size=is_early.sum())

    # --- Incremental payment (D3-D7) ------------------------------------------
    # Early payers are far likelier to pay again; a small trickle of first-time
    # payers also converts. Both depend on latent engagement, so the behavioural
    # features retain signal for the increment specifically.
    late_score = 1.5 * z + 0.9 * df.iap_ctr.to_numpy() + 0.25 * rng.normal(0, 1, n)
    p_late = np.where(
        is_early,
        expit(_solve_intercept(late_score[is_early], P_LATE_GIVEN_EARLY) + late_score),
        expit(_solve_intercept(late_score[~is_early], P_LATE_GIVEN_NONE) + late_score),
    )
    is_late = rng.random(n) < p_late

    delta = np.zeros(n)
    # Whales: a small subset draws from a much heavier tail.
    whale = is_late & (rng.random(n) < 0.03)
    normal_late = is_late & ~whale
    delta[normal_late] = rng.lognormal(1.5, 1.0, normal_late.sum())
    delta[whale] = rng.lognormal(4.2, 1.0, whale.sum())

    df["ltv2"] = ltv2.round(2)
    df["delta_ltv7"] = delta.round(2)
    df["ltv7"] = (df.ltv2 + df.delta_ltv7).round(2)
    df["is_whale"] = whale

    return df.sort_values("install_date").reset_index(drop=True)


def _solve_intercept(score: np.ndarray, target_rate: float) -> float:
    """Bisect for the intercept giving ``mean(expit(intercept + score))`` == target."""
    lo, hi = -40.0, 40.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if expit(mid + score).mean() < target_rate:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def walk_forward_windows(
    df: pd.DataFrame, train_days: int = 60, test_days: int = 30, step_days: int = 30
) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """Rolling train/test splits where the test period strictly follows training.

    Random splits leak the future into training when behaviour is driven by
    releases, events and seasonality, which inflates every offline metric.
    Successive windows overlap by ``train_days - step_days``, so the windows are
    not fully independent and confidence intervals are mildly understated. That
    is a deliberate trade-off under a limited observation period.
    """
    out = []
    t0 = df.install_date.min()
    end = df.install_date.max()
    while True:
        train_end = t0 + pd.Timedelta(days=train_days)
        test_end = train_end + pd.Timedelta(days=test_days)
        if test_end > end:
            break
        out.append(
            (
                df[(df.install_date >= t0) & (df.install_date < train_end)],
                df[(df.install_date >= train_end) & (df.install_date < test_end)],
            )
        )
        t0 = t0 + pd.Timedelta(days=step_days)
    return out
