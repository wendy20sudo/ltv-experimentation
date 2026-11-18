"""Demonstration of label leakage in LTV prediction.

The failure mode: predict ``ltv7`` while ``ltv2`` sits in the feature set.
Because ``ltv7 = ltv2 + delta_ltv7`` and the overwhelming majority of payers
never pay again, copying ``ltv2`` forward is right almost every time. The model
scores beautifully offline and is worthless in production -- it has learned to
transcribe a column, not to predict anything.

Two things are worth separating here, because only one of them is the real fix:

1. **Dropping the leaked feature.** Necessary, not sufficient. The target still
   contains ``ltv2``, a large block already known at scoring time.
2. **Redefining the target** as ``delta_ltv7 = ltv7 - ltv2``. This is the fix.
   The business question is "will they spend more, and how much" -- the unknown
   part. Predicting a total that is mostly known inflates every metric and
   obscures whether the model has any signal at all.

The general lesson: a model that beats expectations deserves more suspicion
than one that misses them. A bad result gets investigated; a good one gets
shipped.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score

from synth import FEATURES


def _fit_eval(train, test, features, target) -> dict:
    model = HistGradientBoostingRegressor(
        max_depth=6, max_iter=150, learning_rate=0.1, random_state=0
    ).fit(train[features], train[target])
    pred = model.predict(test[features])
    actual = test[target].to_numpy()
    denom = np.abs(actual).sum()
    return {
        "MAE": mean_absolute_error(actual, pred),
        "R2": r2_score(actual, pred),
        "WMAPE": np.abs(pred - actual).sum() / denom if denom else np.nan,
    }


def leakage_demo(df: pd.DataFrame, train_frac: float = 0.7) -> pd.DataFrame:
    """Compare the leaked setup, the partial fix, and the correct target.

    Reported on two populations, because which one you pick changes the verdict:

    * **All users.** 99.5% never pay, so ``ltv7 == ltv2 == 0`` for almost
      everyone and the leaked model posts a near-zero MAE. This is the number
      that gets a model shipped. The copy-forward row shows it is achievable
      with no model at all.
    * **Payers only.** The population the model exists to rank. Here the leaked
      model's advantage shrinks to roughly what copying the column already
      gives you.

    Choosing the evaluation population is a modelling decision, not an
    afterthought. A headline metric computed over a population that is 99.5%
    trivial cases is not evidence of anything.
    """
    df = df.sort_values("install_date")
    cut = int(len(df) * train_frac)
    train, test = df.iloc[:cut], df.iloc[cut:]
    train_p = train[train.ltv2 > 0]

    def copy_forward(t: pd.DataFrame) -> dict:
        actual, copy = t.ltv7.to_numpy(), t.ltv2.to_numpy()
        return {
            "MAE": float(np.abs(copy - actual).mean()),
            "R2": r2_score(actual, copy),
            "WMAPE": float(np.abs(copy - actual).sum() / np.abs(actual).sum()),
        }

    rows = []
    for pop, tr, te in (
        ("all users", train, test),
        ("payers only", train_p, test[test.ltv2 > 0]),
    ):
        rows += [
            {"population": pop, "setup": "copy ltv2 forward (no model)", **copy_forward(te)},
            {"population": pop, "setup": "ltv2 IN features -> ltv7  [LEAKED]",
             **_fit_eval(tr, te, FEATURES + ["ltv2"], "ltv7")},
            {"population": pop, "setup": "ltv2 dropped     -> ltv7  [partial]",
             **_fit_eval(tr, te, FEATURES, "ltv7")},
            {"population": pop, "setup": "ltv2 dropped     -> delta [correct]",
             **_fit_eval(tr, te, FEATURES, "delta_ltv7")},
        ]
    return pd.DataFrame(rows)
