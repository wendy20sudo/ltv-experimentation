#!/usr/bin/env python3
"""Reproduce every result in the README.

    python run_all.py              # full run (~1 min)
    python run_all.py --n 400000   # quick run, noisier numbers
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import numpy as np
import pandas as pd

from experiments import (
    required_n_per_arm,
    simulate_cuped,
    simulate_peeking,
    srm_check,
)
from leakage import leakage_demo
from synth import generate, walk_forward_windows
from two_stage import compare_single_model, evaluate_window


def header(text: str) -> None:
    print(f"\n{'=' * 78}\n{text}\n{'=' * 78}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1_800_000)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", lambda v: f"{v:,.3f}")

    header("1. Synthetic data")
    df = generate(n=args.n, seed=args.seed)
    payers = df.ltv2 > 0
    print(f"installs                    {len(df):,}")
    print(f"early payers (ltv2>0)       {payers.sum():,}  ({payers.mean():.3%})")
    print(f"any D3-D7 increment         {(df.delta_ltv7>0).mean():.3%} of installs")
    print(f"increment among payers      {(df.loc[payers,'delta_ltv7']>0).mean():.1%}")
    print(f"whales                      {df.is_whale.sum():,} users, "
          f"{df.loc[df.is_whale,'delta_ltv7'].sum()/df.delta_ltv7.sum():.1%} of increment revenue")

    header("2. Label leakage")
    print(leakage_demo(df).to_string(index=False))

    header("3. Two-stage model, walk-forward")
    windows = walk_forward_windows(df)
    rows, sweeps = [], []
    for i, (train, test) in enumerate(windows, 1):
        r = evaluate_window(train, test)
        rows.append({
            "window": f"W{i}",
            "test_users": r["n_test"],
            "payers": r["n_payers"],
            "auc_full": r["auc_full"],
            "auc_payers": r["auc_payers"],
            "ap_payers": r["ap_payers"],
            "mace": r["mace_full"],
            "tot_err_uncal": r["total_error_uncal"],
            "tot_err_cal": r["total_error_cal"],
            "tot_err_naive": r["total_error_naive"],
            "wmape_cal": r["wmape_cal"],
            "wmape_naive": r["wmape_naive"],
        })
        s = r["sweep_payers"].copy()
        s.insert(0, "window", f"W{i}")
        sweeps.append(s)
    print(pd.DataFrame(rows).to_string(index=False))

    header("4. Operating point: precision and recall move together")
    sweep = pd.concat(sweeps)
    print(sweep.groupby("threshold")[["precision", "recall", "f1"]].mean().to_string())
    print(
        "\nPrecision and recall must be quoted from the SAME row. A claim pairing\n"
        "recall from one threshold with precision from another describes a model\n"
        "that does not exist."
    )

    header("5. Two-stage vs single regression, at matched recall")
    tr, te = windows[0]
    single = compare_single_model(tr, te)
    two = sweeps[0]
    for target in (0.95, 0.85, 0.70):
        s = single.iloc[(single.recall - target).abs().argsort().iloc[0]]
        t = two.iloc[(two.recall - target).abs().argsort().iloc[0]]
        print(f"  recall~{target:.0%}:  single P={s.precision:.3f} (r={s.recall:.3f})   "
              f"two-stage P={t.precision:.3f} (r={t.recall:.3f})")
    print("\nThe two-stage advantage is tunability and clean zeros, not raw precision.")

    header("6. Experiment hygiene")
    print("SRM, healthy split:", {k: round(v, 4) if isinstance(v, float) else v
                                  for k, v in srm_check(50_012, 49_988).items()})
    print("SRM, broken split: ", {k: round(v, 6) if isinstance(v, float) else v
                                  for k, v in srm_check(51_500, 48_500).items()})
    print(f"\nUsers/arm to detect +3% on a 5% baseline:  {required_n_per_arm(0.05, 0.03):,}")
    print(f"Users/arm to detect +10% on a 5% baseline: {required_n_per_arm(0.05, 0.10):,}")

    header("7. CUPED")
    print(simulate_cuped().to_string(index=False))

    header("8. Peeking")
    print(simulate_peeking(n_sims=1000).to_string(index=False))
    print("\nTen interim looks at a nominal 5% test produce a far higher real\n"
          "false positive rate. The confidence sequence stays valid throughout,\n"
          "at the cost of being conservative.")


if __name__ == "__main__":
    main()
