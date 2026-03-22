#!/usr/bin/env python3
from pathlib import Path
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


REQUIRED_COLS = ["seconds_elapsed", "x", "y", "z"]


def load_sensor_csv(csv_path: Path):
    df = pd.read_csv(csv_path)

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"{csv_path.name}: Missing required columns {missing}. "
            f"Found columns: {list(df.columns)}"
        )

    for c in REQUIRED_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=REQUIRED_COLS).reset_index(drop=True)
    return df


def inspect_sensor_df(df: pd.DataFrame, sensor_name: str, csv_path: Path):
    if df.empty:
        return {
            "sensor_name": sensor_name,
            "csv_path": csv_path,
            "samples": 0,
            "duration": 0.0,
            "avg_dt": np.nan,
            "median_dt": np.nan,
            "min_dt": np.nan,
            "max_dt": np.nan,
            "avg_rate": np.nan,
            "median_rate": np.nan,
            "monotonic": True,
            "large_gap": False,
            "empty": True,
        }

    t = df["seconds_elapsed"].to_numpy()
    n = len(t)
    duration = float(t[-1] - t[0]) if n > 1 else 0.0

    if n > 1:
        dt = t[1:] - t[:-1]
        avg_dt = float(dt.mean())
        median_dt = float(np.median(dt))
        min_dt = float(dt.min())
        max_dt = float(dt.max())
        avg_rate = 1.0 / avg_dt if avg_dt > 0 else float("nan")
        median_rate = 1.0 / median_dt if median_dt > 0 else float("nan")
        monotonic = bool((dt >= 0).all())
        large_gap = bool(median_dt > 0 and max_dt > 5 * median_dt)
    else:
        avg_dt = median_dt = min_dt = max_dt = 0.0
        avg_rate = median_rate = 0.0
        monotonic = True
        large_gap = False

    return {
        "sensor_name": sensor_name,
        "csv_path": csv_path,
        "samples": n,
        "duration": duration,
        "avg_dt": avg_dt,
        "median_dt": median_dt,
        "min_dt": min_dt,
        "max_dt": max_dt,
        "avg_rate": avg_rate,
        "median_rate": median_rate,
        "monotonic": monotonic,
        "large_gap": large_gap,
        "empty": False,
    }


def print_inspection(stats: dict):
    print("\n==============================")
    print(f"Sensor: {stats['sensor_name']}")
    print(f"File  : {stats['csv_path']}")
    print("==============================")

    if stats["empty"]:
        print("No valid rows found after cleaning CSV.")
        return

    print(f"Samples        : {stats['samples']}")
    print(f"Duration       : {stats['duration']:.3f} sec")

    print("\nSampling statistics")
    print(f"Average dt     : {stats['avg_dt']:.6f} sec")
    print(f"Median dt      : {stats['median_dt']:.6f} sec")
    print(f"Min dt         : {stats['min_dt']:.6f} sec")
    print(f"Max dt         : {stats['max_dt']:.6f} sec")

    print(f"\nAverage rate   : {stats['avg_rate']:.2f} Hz")
    print(f"Median rate    : {stats['median_rate']:.2f} Hz")

    print(f"\nMonotonic time : {stats['monotonic']}")
    if stats["large_gap"]:
        print("Large gap detected in timestamps")


def plot_sensor(ax, df: pd.DataFrame, sensor_name: str, ylabel: str):
    t = df["seconds_elapsed"] - df["seconds_elapsed"].iloc[0]
    ax.plot(t, df["x"], label="x")
    ax.plot(t, df["y"], label="y")
    ax.plot(t, df["z"], label="z")
    ax.set_title(sensor_name)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(ylabel)
    ax.grid(True)
    ax.legend()


def main():
    parser = argparse.ArgumentParser(
        description="Inspect and plot accelerometer, gyroscope, and magnetometer data from one recording folder"
    )
    parser.add_argument(
        "recording_dir",
        type=str,
        help="Path to recording folder containing sensor CSV files",
    )
    parser.add_argument("--title", type=str, default=None, help="Figure title")
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save PNG instead of showing interactive plot",
    )
    args = parser.parse_args()

    recording_dir = Path(args.recording_dir)
    if not recording_dir.exists() or not recording_dir.is_dir():
        raise FileNotFoundError(f"Recording folder not found: {recording_dir}")

    sensor_files = {
        "Accelerometer": recording_dir / "Accelerometer.csv",
        "Gyroscope": recording_dir / "Gyroscope.csv",
        "Magnetometer": recording_dir / "Magnetometer.csv",
    }

    sensor_units = {
        "Accelerometer": "Acceleration",
        "Gyroscope": "Angular velocity",
        "Magnetometer": "Magnetic field",
    }

    loaded = []
    for sensor_name, csv_path in sensor_files.items():
        if not csv_path.exists():
            print(f"Skipping {sensor_name}: file not found ({csv_path})")
            continue

        df = load_sensor_csv(csv_path)
        stats = inspect_sensor_df(df, sensor_name, csv_path)
        print_inspection(stats)

        if stats["empty"]:
            print(f"Skipping plot for {sensor_name}: no valid data")
            continue

        loaded.append((sensor_name, df))

    if not loaded:
        raise RuntimeError("No valid sensor CSV files found in the recording folder.")

    fig, axes = plt.subplots(len(loaded), 1, figsize=(12, 4 * len(loaded)), sharex=False)
    if len(loaded) == 1:
        axes = [axes]

    for ax, (sensor_name, df) in zip(axes, loaded):
        plot_sensor(ax, df, sensor_name, sensor_units[sensor_name])

    fig.suptitle(args.title if args.title else recording_dir.name, fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    if args.save:
        out_path = recording_dir / f"{recording_dir.name}_sensor_plot.png"
        plt.savefig(out_path, dpi=150)
        print(f"\nSaved plot to: {out_path}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
