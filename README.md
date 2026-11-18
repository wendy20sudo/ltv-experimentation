# Predicting LTV without fooling yourself

A small, reproducible study of the things that go wrong when you predict user
lifetime value for ad bidding, and what to do about each one.

Four failure modes, each demonstrated with runnable code on synthetic data:

1. **Label leakage** — a model that scores brilliantly because it learned to copy a column.
2. **Zero inflation** — 99.6% of users are worth nothing, which breaks naive regression.
3. **Miscalibration** — a model whose ranking is good and whose numbers are wrong.
4. **Peeking** — a correctly built experiment, invalidated by checking it too often.

```bash
pip install -r requirements.txt
python run_all.py          # ~1 min, reproduces every table below
python run_all.py --n 400000   # faster, noisier
```

No real or proprietary data is used anywhere. `src/synth.py` generates everything.

---

## The setting

Smart bidding learns from the conversion signals you send it. If all you can
return is "installed" or "registered", a user who will never pay and a user who
will spend hundreds look identical, so the bidder optimises for the cheapest
installs — which are the least valuable users.

Actual payment value is long-tailed and arrives over weeks. A bidding model's
learning window is days. **That gap is the whole reason predicted LTV exists:**
use early behaviour to predict long-run value, and return the prediction as a
fast proxy for the slow truth.

The synthetic data reproduces the three properties that make this hard:
0.5% of installs pay at all, half of those pay again in D3–D7, and 201 "whales"
out of 1.8M users account for **33% of all incremental revenue**.

---

## 1. Label leakage

Predict `ltv7` with `ltv2` in the feature set. Since `ltv7 = ltv2 + delta`, and
almost nobody pays twice, copying one column is right nearly every time.

| Population | Setup | MAE | WMAPE |
|---|---|---|---|
| all users | copy `ltv2` forward — **no model at all** | **0.045** | 0.482 |
| all users | `ltv2` in features → predict `ltv7` **[leaked]** | 0.090 | 0.973 |
| all users | `ltv2` dropped → predict `ltv7` | 0.171 | 1.844 |
| all users | `ltv2` dropped → predict `delta` **[correct]** | 0.081 | 1.823 |

The leaked model looks twice as accurate as the honest one. It is also **beaten
by copying a column**. Its apparent skill is entirely the leak.

**Dropping the leaked feature is not the fix.** The target still contains
`ltv2`, a large block already known at scoring time. The fix is redefining the
target as `delta_ltv7 = ltv7 − ltv2` — the part that is actually unknown, and
the part the business is asking about.

The general lesson: *a result that beats expectations deserves more suspicion
than one that misses.* A bad result gets investigated. A good one gets shipped.

---

## 2. Two-stage model, walk-forward validation

99.6% of users have zero increment, so a single regression is dominated by
zeros and its cheapest strategy is to predict ~0 everywhere. Stage 1 classifies
*whether*; Stage 2 predicts *how much*, trained only on genuine payers.

Validation is walk-forward — three windows, each tested on a period strictly
after its training data. Random splits leak the future into training when
behaviour is driven by releases and seasonality.

| Window | Test users | Payers | AUC (all) | AUC (payers) | AP | Total error, uncal. | **Total error, calibrated** | Naive |
|---|---|---|---|---|---|---|---|---|
| W1 | 299,737 | 1,499 | 0.882 | 0.770 | 0.758 | +24.6% | **−9.5%** | −41.0% |
| W2 | 299,989 | 1,454 | 0.888 | 0.778 | 0.791 | +32.2% | **−0.5%** | −37.9% |
| W3 | 299,609 | 1,578 | 0.884 | 0.764 | 0.775 | +21.4% | **−6.7%** | −40.7% |

The split carries a **7-day embargo**: the label is a D7 outcome, so standing at
the training cutoff you do not yet know how users who installed in the last
week turned out. Training on them would use labels that could not exist at the
time — a quieter version of the same temporal leakage the walk-forward split
exists to prevent.

Stage 2 over-predicts by 21–32% before calibration, because it trains only on
payers but scores everyone Stage 1 flags — around 45% of whom never pay.
Isotonic regression corrects it. Isotonic rather than Platt because Platt
assumes a sigmoid-shaped bias and this one is not sigmoid-shaped; isotonic
assumes only monotonicity.

Two implementation details decide whether the calibration works at all, and
both took debugging to find:

- **Calibrate on the population you score.** Fitting the calibrator on all users
  means fitting on a sample that is 99.6% zeros; it learns to shrink every
  prediction to nothing.
- **Use out-of-fold predictions.** Increment-positive users are in Stage 2's
  training set, so in-sample predictions for them are optimistic. Fit on those
  and isotonic concludes less correction is needed than really is.

---

## 3. Precision and recall come from the same row

| Threshold | Precision | Recall | F1 |
|---|---|---|---|
| 0.15 | 0.552 | 0.972 | 0.704 |
| 0.20 | 0.569 | 0.953 | 0.712 |
| 0.30 | 0.600 | 0.911 | 0.723 |
| 0.40 | 0.632 | 0.876 | 0.734 |
| 0.50 | 0.658 | 0.824 | 0.732 |
| **0.70** | **0.710** | **0.671** | 0.690 |
| 0.80 | 0.760 | 0.547 | 0.636 |

Moving from 0.20 to 0.70 raises precision from 0.57 to 0.71 and drops recall
from 0.95 to 0.67. That is a real, defensible trade-off: fewer false positives
into Stage 2, at the cost of missing roughly a third of genuine payers.

What it is *not* is "precision 0.72 at recall 0.96". Quoting a precision from
one operating point beside a recall from another describes a model that does
not exist, and one question — *what was recall at that precision?* — exposes it.

Which point to pick is a business question, not a modelling one. Returning too
few high-value conversions starves the bidder of the signal it learns from, so
high recall is right while buying volume; stricter thresholds are right when
controlling cost.

### Is two-stage actually better than one model?

Compared honestly — at matched recall rather than at each model's own default
threshold — much less than the usual framing suggests:

| Recall | Single regression | Two-stage |
|---|---|---|
| ~95% | 0.538 | 0.537 |
| ~85% | 0.625 | 0.637 |
| ~70% | 0.681 | 0.694 |

The advantage is real but modest, and it is structural rather than numerical:
Stage 1 emits a probability you can move along the curve, and Stage 2 never
produces output for users predicted to be worth nothing.

---

## 4. Calibration is not accuracy

Stage 1's probabilities are badly calibrated — mean absolute calibration error
0.24. The ranking is trustworthy (AUC 0.89); the numbers are not. A model
emitting 0.6 is nowhere near 60% likely to be right.

This matters precisely because of what consumes the output. A bidder receiving
0.3 multiplies it by a value to compute a bid, so it needs the magnitude to be
true, not just the order. **AUC cannot tell you this.** Any downstream use of
Stage 1's probability as a probability needs a calibration layer first.

---

## 5. Experiment hygiene

**Sample ratio mismatch.** A 51,500 / 48,500 split tests at p < 1e−6 against an
intended 50/50. Assignment is broken; nothing downstream is interpretable. Check
before reading results, not after.

**Power.** Detecting a +3% relative lift on a 5% baseline needs **336,102 users
per arm**. At +10% it needs 31,234. An underpowered test that comes back flat
has not shown the absence of an effect — but it is routinely reported that way.

**CUPED.** A pre-experiment covariate correlated ρ=0.7 with the outcome:

| Method | Effect | Std. error | Variance reduction |
|---|---|---|---|
| Raw | +2.12 | 0.201 | — |
| CUPED | +2.18 | 0.142 | **49.4%** |

Unbiased, because the covariate is measured before assignment and cannot be
affected by treatment. Variance falls by about ρ², here halving it — equivalent
to doubling the sample size, for free.

**Peeking.** Ten interim looks at a nominal 5% test, under A/A where every
positive is false:

| Method | False positive rate |
|---|---|
| Fixed-horizon, 10 peeks | **18.6%** |
| Always-valid confidence sequence | 0.1% |

Nearly four times the nominal rate. The confidence sequence holds at every
sample size simultaneously, so it can be monitored continuously — at the cost
of being conservative, visibly so here.

---

## Does it hold across seeds?

A result that only works on one random seed is not a result. Across 4 seeds ×
3 windows:

| | |
|---|---|
| Calibration reduced absolute bias in | **11 of 12** window-seeds |
| Mean total error, uncalibrated | +27.1% |
| Mean total error, calibrated | **−1.3%** |
| Sign of residual bias | 5 positive, 7 negative |

The balanced sign matters: it says the remaining error is noise, not a
systematic over-correction. The single failure is instructive — uncalibrated
bias there was already only +6.7%, and when there is little bias to remove,
calibration contributes variance without much to correct.

```bash
python tests/test_core.py    # 12 invariant tests, no pytest needed
pytest -q                    # or via pytest
```

The tests assert reasoning, not just execution: that `ltv7 == ltv2 + delta`
exactly, that no training install falls within the label horizon of the test
start, that copying a column beats the leaked model, that precision rises
monotonically as recall falls, that CUPED's θ matches its closed form, and
that peeking inflates the false positive rate above nominal.

## Limitations

Stated because a portfolio piece that hides its weaknesses is worth less than
one that doesn't.

- **WMAPE is worse than the naive baseline** (0.46–0.49 vs 0.38–0.41). The model
  buys large gains in *aggregate* bias while per-user absolute error gets worse
  than simply predicting zero. This is usable for channel- and cohort-level
  revenue planning; it is **not** usable for per-user pricing.
- **Whales stay underestimated.** Isotonic cannot predict above the largest
  value in its calibration set, so the heaviest spenders are systematically
  clipped. That residual is most of the remaining −2% to −11% negative bias.
- **Windows overlap.** Successive training periods share 30 days, so the three
  windows are not independent and confidence intervals are mildly understated.
- **The always-valid bound is very conservative** at these sample sizes (0.1%
  against a nominal 5%). Tightening it means tuning the mixture parameter.
- **Synthetic data is kind.** No seasonality, no release effects, no bot
  traffic, no attribution error, stationary behaviour. Real data has all five.

## Layout

```
src/synth.py        synthetic data + walk-forward splits with label embargo
src/leakage.py      label leakage demonstration
src/two_stage.py    two-stage model, isotonic calibration, metrics
src/experiments.py  SRM, power, CUPED, sequential testing
run_all.py          reproduces every table above
tests/test_core.py  12 invariant tests
```
