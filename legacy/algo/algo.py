import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json, argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from model import load_and_join, build_df, train_model, FEAT


def write_jsonl(p, rows):
    with open(p, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", required=True, help="Training delivery CSV (with actual_time)")
    ap.add_argument("--input_csv", required=True, help="Inference delivery CSV")
    ap.add_argument("--weather_csv", required=True, help="Weather CSV (weather2.csv)")
    ap.add_argument("--out_jsonl", default="out.jsonl")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    print("Loading training data...")
    tr_merged = load_and_join(args.train_csv, args.weather_csv)
    tr_df = build_df(tr_merged, require_a_time=True)
    print(f"  {len(tr_df)} training rows")

    print("Loading inference data...")
    te_merged = load_and_join(args.input_csv, args.weather_csv)
    te_df = build_df(te_merged, require_a_time=False)
    print(f"  {len(te_df)} inference rows")

    # Align categories between train and inference
    for col in ['source_center', 'dest_center']:
        all_cats = tr_df[col].cat.categories.union(te_df[col].cat.categories)
        tr_df[col] = tr_df[col].cat.set_categories(all_cats)
        te_df[col] = te_df[col].cat.set_categories(all_cats)

    # Train / validation split
    tr, va = train_test_split(tr_df, test_size=0.2, random_state=args.seed)

    print("Training model...")
    model = train_model(tr, seed=args.seed)

    # Validation residuals
    va_pred = model.predict(va[FEAT])
    res = va["a_time"].astype(float).values - va_pred

    s = va[["source_center", "dest_center", "hour"]].copy()
    s["res"] = res

    stats = (
        s.groupby(["source_center", "dest_center", "hour"])["res"]
        .agg(
            p_late=lambda r: float(np.mean(r > 0)),
            e_delay=lambda r: float(np.mean(np.clip(r, 0, None))),
            sigma=lambda r: float(np.std(r, ddof=0)),
        )
        .reset_index()
    )

    # Test prediction
    te_pred = model.predict(te_df[FEAT])

    out = te_df[
        ["shipment_id", "ts", "source_center", "dest_center", "hour"]
    ].copy()
    out["e_time"] = te_pred

    out = out.merge(
        stats,
        on=["source_center", "dest_center", "hour"],
        how="left"
    )

    # Global fallback for unseen buckets
    gp = float(stats["p_late"].mean()) if len(stats) else 0.0
    gd = float(stats["e_delay"].mean()) if len(stats) else 0.0
    gs = float(stats["sigma"].mean()) if len(stats) else 1.0

    out["p_late"] = out["p_late"].fillna(gp)
    out["e_delay"] = out["e_delay"].fillna(gd)
    out["sigma"] = out["sigma"].fillna(gs)

    out["confidence"] = 1.0 / (
        1.0 + np.maximum(out["sigma"].values, 1e-6)
    )

    rows = [
        {
            "confidence": float(r.confidence),
            "shipment_id": r.shipment_id,
            "ts": str(r.ts),
            "p_late": float(r.p_late),
            "e_delay": float(r.e_delay),
            "e_time": float(r.e_time),
        }
        for r in out.itertuples(index=False)
    ]

    write_jsonl(args.out_jsonl, rows)

    print(f"[OK] wrote {args.out_jsonl} (n={len(rows)})")
    print(rows[:3])

if __name__ == "__main__":
    main()
