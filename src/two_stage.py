"""Two-stage zero-inflated pLTV model with isotonic calibration.

Why two stages rather than one regression:

* 99.6% of installs have zero D3-D7 increment, so a single regression is
  dominated by zeros and its cheapest strategy is to predict ~0 everywhere.
* The paying tail is long, producing large unstable gradients during training.

Splitting "will they pay again" from "how much" gives Stage 1 a probability that
can be moved along the precision/recall curve to suit the business, and stops
Stage 2 emitting noise for users predicted to be worth nothing.

A caveat worth stating plainly: the two-stage structure is not automatically
better than one model at every operating point. Compared fairly -- at matched
recall, or on a threshold-free measure like AUC-PR -- a tuned single regression
is competitive. The structural advantages are tunability and clean zeros, not a
headline precision number. ``compare_single_model`` makes that comparison.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, roc_auc_score

from synth import FEATURES


class TwoStageLTV:
    """Stage 1 classifies "any increment"; Stage 2 predicts its size."""

    def __init__(self, threshold: float = 0.20, seed: int = 0):
        self.threshold = threshold
        self.seed = seed
        # ``class_weight="balanced"`` is the sklearn equivalent of XGBoost's
        # scale_pos_weight. Without it a 0.4% base rate keeps every predicted
        # probability far below 0.5 and the threshold becomes meaningless.
        # The cost is that probabilities come out systematically overconfident,
        # which is exactly what ``mace`` measures below.
        self.stage1 = HistGradientBoostingClassifier(
            max_depth=6, max_iter=200, learning_rate=0.1,
            class_weight="balanced", random_state=seed,
        )
        self.stage2 = HistGradientBoostingRegressor(
            max_depth=6, max_iter=200, learning_rate=0.1, random_state=seed
        )
        self.calibrator: IsotonicRegression | None = None

    def fit(self, train: pd.DataFrame, n_folds: int = 5) -> "TwoStageLTV":
        """Fit both stages and calibrate Stage 2 on out-of-fold predictions."""
        train = train.sort_values("install_date")

        self.stage1.fit(train[FEATURES], (train.delta_ltv7 > 0).astype(int))

        # Stage 2 trains only on users who genuinely had an increment: the
        # distribution stays clean instead of being swamped by zeros.
        pos = train[train.delta_ltv7 > 0]
        self.stage2.fit(pos[FEATURES], pos.delta_ltv7)

        self._fit_calibrator(train, pos, n_folds)
        return self

    def _fit_calibrator(self, train: pd.DataFrame, pos: pd.DataFrame, n_folds: int) -> None:
        """Map raw Stage 2 output onto realised increments.

        Stage 2 trains only on genuine payers but is scored on everyone Stage 1
        flags -- a third of whom never pay. That distribution shift makes it
        over-predict in aggregate. Isotonic regression is the right correction:
        it assumes only monotonicity, where Platt scaling would impose a sigmoid
        shape the bias does not have.

        Two details that decide whether this works at all:

        * **Restrict to the operating population** (``ltv2 > 0``). Calibrating
          on everyone means calibrating on a sample that is ~99.6% zeros, and
          the calibrator learns to shrink every prediction to nothing.
        * **Use out-of-fold Stage 2 predictions.** Increment-positive users are
          in Stage 2's training set, so in-sample predictions for them are
          optimistic. Fit on those and isotonic concludes less correction is
          needed than really is, and the model keeps over-predicting on unseen
          users. Zero-increment users are unaffected -- they were never in
          Stage 2's training data -- so only the positives need folding out.
        """
        payers = train[train.ltv2 > 0]
        proba = self.stage1.predict_proba(payers[FEATURES])[:, 1]
        flagged = payers[proba >= self.threshold]
        if len(flagged) < 100:
            self.calibrator = None
            return

        oof = np.empty(len(flagged))
        is_pos = (flagged.delta_ltv7 > 0).to_numpy()

        # Zero-increment users: the fitted model never saw them.
        if (~is_pos).sum():
            oof[~is_pos] = self.stage2.predict(flagged.loc[~is_pos, FEATURES])

        # Increment-positive users: predict each from a model trained without it.
        if is_pos.sum():
            pos_idx = pos.index
            fold_of = pd.Series(
                np.arange(len(pos_idx)) % n_folds, index=pos_idx
            )
            target = flagged.loc[is_pos]
            preds = np.empty(len(target))
            for k in range(n_folds):
                hold = target.index[fold_of.reindex(target.index).to_numpy() == k]
                if len(hold) == 0:
                    continue
                tr_k = pos.loc[fold_of[fold_of != k].index]
                m = HistGradientBoostingRegressor(
                    max_depth=6, max_iter=200, learning_rate=0.1, random_state=self.seed
                ).fit(tr_k[FEATURES], tr_k.delta_ltv7)
                preds[target.index.get_indexer(hold)] = m.predict(
                    target.loc[hold, FEATURES]
                )
            oof[is_pos] = preds

        self.calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0)
        self.calibrator.fit(oof, flagged.delta_ltv7.to_numpy())

    def predict_delta(self, df: pd.DataFrame, calibrated: bool = True) -> np.ndarray:
        """Predicted D3-D7 increment; zero wherever Stage 1 says no."""
        proba = self.stage1.predict_proba(df[FEATURES])[:, 1]
        out = np.zeros(len(df))
        flag = proba >= self.threshold
        if flag.sum() == 0:
            return out
        raw = self.stage2.predict(df.loc[flag, FEATURES])
        if calibrated and self.calibrator is not None:
            raw = self.calibrator.predict(raw)
        out[flag] = np.clip(raw, 0, None)
        return out

    def stage1_proba(self, df: pd.DataFrame) -> np.ndarray:
        return self.stage1.predict_proba(df[FEATURES])[:, 1]


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------

def total_error(pred_delta: np.ndarray, test: pd.DataFrame) -> float:
    """Bias of predicted total LTV7 revenue. Target is 0%.

    Evaluated on total revenue (``ltv2 + delta``) rather than the increment
    alone, because that is the quantity a bidding or budgeting system consumes.
    """
    pred_total = test.ltv2.to_numpy() + pred_delta
    actual = test.ltv7.to_numpy()
    return float((pred_total.sum() - actual.sum()) / actual.sum())


def wmape(pred_delta: np.ndarray, test: pd.DataFrame) -> float:
    """Revenue-weighted absolute error at user level."""
    pred_total = test.ltv2.to_numpy() + pred_delta
    actual = test.ltv7.to_numpy()
    return float(np.abs(pred_total - actual).sum() / actual.sum())


def mace(proba: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Mean absolute calibration error of Stage 1 probabilities.

    AUC says whether the ranking is right. MACE says whether a 0.3 means 30%.
    A bidder multiplying a probability by a value needs the second property;
    ranking alone is not enough.
    """
    edges = np.quantile(proba, np.linspace(0, 1, bins + 1))
    edges = np.unique(edges)
    if len(edges) < 3:
        return float("nan")
    idx = np.clip(np.digitize(proba, edges[1:-1]), 0, len(edges) - 2)
    errs, weights = [], []
    for b in range(len(edges) - 1):
        m = idx == b
        if m.sum() == 0:
            continue
        errs.append(abs(proba[m].mean() - actual[m].mean()))
        weights.append(m.sum())
    return float(np.average(errs, weights=weights))


def threshold_sweep(proba: np.ndarray, label: np.ndarray, grid=None) -> pd.DataFrame:
    """Precision / recall / F1 across thresholds.

    Precision and recall come from the *same* threshold on every row. Quoting a
    precision from one operating point beside a recall from another describes a
    model that does not exist.
    """
    grid = grid if grid is not None else [0.15, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]
    rows = []
    for t in grid:
        pred = proba >= t
        tp = int((pred & (label == 1)).sum())
        fp = int((pred & (label == 0)).sum())
        fn = int((~pred & (label == 1)).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        rows.append(
            {"threshold": t, "precision": prec, "recall": rec, "f1": f1,
             "flagged": int(pred.sum()), "false_positives": fp}
        )
    return pd.DataFrame(rows)


def evaluate_window(train: pd.DataFrame, test: pd.DataFrame, threshold: float = 0.20) -> dict:
    """Fit on ``train`` and report the metrics that matter on ``test``.

    Headline figures are computed on the ``ltv2 > 0`` subset. On the full
    population a model predicting near-zero everywhere looks excellent and is
    useless; the payer subset is also the group a UA team actually targets.
    """
    model = TwoStageLTV(threshold=threshold, seed=0).fit(train)

    proba = model.stage1_proba(test)
    label = (test.delta_ltv7 > 0).astype(int).to_numpy()
    payer = test.ltv2.to_numpy() > 0

    # Score the operating population only. Scoring everyone and subsetting
    # afterwards gives the same numbers here, but stating it this way keeps the
    # code honest about who the model is actually for.
    sub = test[payer]
    pred_cal = model.predict_delta(sub, calibrated=True)
    pred_raw = model.predict_delta(sub, calibrated=False)
    return {
        "n_test": len(test),
        "n_payers": int(payer.sum()),
        "positive_rate_full": float(label.mean()),
        "positive_rate_payers": float(label[payer].mean()),
        "auc_full": roc_auc_score(label, proba),
        "auc_payers": roc_auc_score(label[payer], proba[payer]),
        "ap_payers": average_precision_score(label[payer], proba[payer]),
        "mace_full": mace(proba, label),
        "total_error_uncal": total_error(pred_raw, sub),
        "total_error_cal": total_error(pred_cal, sub),
        "wmape_uncal": wmape(pred_raw, sub),
        "wmape_cal": wmape(pred_cal, sub),
        "total_error_naive": total_error(np.zeros(len(sub)), sub),
        "wmape_naive": wmape(np.zeros(len(sub)), sub),
        "sweep_payers": threshold_sweep(proba[payer], label[payer]),
        "_model": model,
    }


def compare_single_model(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """Single regression vs two-stage, compared at matched recall.

    The naive comparison -- two-stage precision at its threshold against the
    single model's "predict > 0" -- is not a like-for-like test, because those
    are different operating points. Sweeping the single model's cutoff until
    recall matches is the fair version, and it is usually much closer than the
    naive comparison suggests.
    """
    reg = HistGradientBoostingRegressor(
        max_depth=6, max_iter=200, learning_rate=0.1, random_state=0
    ).fit(train[FEATURES], train.delta_ltv7)
    pred = reg.predict(test[FEATURES])

    payer = test.ltv2.to_numpy() > 0
    label = (test.delta_ltv7 > 0).astype(int).to_numpy()[payer]
    pred_p = pred[payer]

    rows = []
    for cut in np.quantile(pred_p, np.linspace(0.05, 0.95, 19)):
        flag = pred_p >= cut
        tp = int((flag & (label == 1)).sum())
        fp = int((flag & (label == 0)).sum())
        fn = int((~flag & (label == 1)).sum())
        if tp + fp == 0 or tp + fn == 0:
            continue
        rows.append(
            {"cutoff": float(cut), "precision": tp / (tp + fp), "recall": tp / (tp + fn)}
        )
    return pd.DataFrame(rows)
