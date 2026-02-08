import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from model import load_and_join, build_df, train_model, FEAT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--delivery_csv", required=True, help="Delivery CSV (data/delivery.csv)")
    ap.add_argument("--weather_csv", required=True, help="Weather CSV (data/weather2.csv)")
    ap.add_argument("--cutoff_ts", required=True, help="Cutoff timestamp (e.g. 2018-09-30)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_plot_points", type=int, default=800)
    args = ap.parse_args()

    print("Loading and joining data...")
    merged = load_and_join(args.delivery_csv, args.weather_csv)
    df = build_df(merged)
    print(f"  {len(df)} rows after join")

    cutoff = pd.to_datetime(args.cutoff_ts)

    train_df = df[df['ts'] <= cutoff].copy()
    test_df = df[df['ts'] > cutoff].copy()

    if len(train_df) < 10:
        raise RuntimeError(f"Training set too small: {len(train_df)} rows")
    if len(test_df) < 9:
        raise RuntimeError(f"Test set too small: {len(test_df)} rows")

    print(f"Train: {len(train_df)} rows, Test: {len(test_df)} rows")

    model = train_model(train_df, seed=args.seed)

    pred = model.predict(test_df[FEAT])
    y = test_df["a_time"].astype(float).values

    err = y - pred
    mse = float(np.mean(err**2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(err)))
    print(f"cutoff={args.cutoff_ts}  train_n={len(train_df)}  test_n={len(test_df)}")
    print(f"MSE={mse:.6f}  RMSE={rmse:.6f}  MAE={mae:.6f}")

    # --- prepare plot data (downsample if too many points) ---
    plot_df = test_df[["ts"]].copy()
    plot_df["actual"] = y
    plot_df["pred"] = pred

    if len(plot_df) > args.max_plot_points:
        idx = np.linspace(0, len(plot_df) - 1, args.max_plot_points).astype(int)
        plot_df = plot_df.iloc[idx].copy()

    plt.figure()
    plt.plot(plot_df["ts"], plot_df["actual"], label="Actual")
    plt.plot(plot_df["ts"], plot_df["pred"], label="Predicted")
    plt.xlabel("Time")
    plt.ylabel("a_time")
    plt.title("Actual vs Predicted after cutoff")
    plt.legend()
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
