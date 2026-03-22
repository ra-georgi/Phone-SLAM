#!/usr/bin/env python3

import argparse
import csv
import json
import shutil
import subprocess
from pathlib import Path


def run_ffprobe_json(cmd):
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return json.loads(result.stdout)


def get_frame_pts(video_path: Path):
    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "frame=pts_time",
        "-of", "json",
        str(video_path),
    ]
    data = run_ffprobe_json(cmd)

    pts_list = []
    for frame in data.get("frames", []):
        pts = frame.get("pts_time")
        if pts is None:
            continue
        pts_list.append(float(pts))

    if not pts_list:
        raise RuntimeError("No frame PTS values found in video")

    return pts_list


def parse_video_start_epoch_ns(video_path: Path):
    stem = video_path.stem
    if not stem.isdigit():
        raise ValueError(
            f"Expected video filename stem to be all digits, got: {stem}"
        )

    video_start_ms = int(stem)
    return video_start_ms * 1_000_000


def count_csv_rows(csv_path: Path) -> int:
    with open(csv_path, "r", newline="") as f:
        reader = csv.reader(f)
        next(reader, None)
        return sum(1 for _ in reader)


def extract_frames(video_path: Path, output_dir: Path, image_format: str):
    frame_pattern = output_dir / f"frame_%06d.{image_format}"
    cmd = ["ffmpeg", "-y", "-i", str(video_path), str(frame_pattern)]

    print("\nRunning ffmpeg:")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)

    frames = sorted(output_dir.glob(f"frame_*.{image_format}"))
    return frames


def write_timestamps_csv(video_path: Path, output_csv: Path, frame_pts_list):
    video_start_epoch_ns = parse_video_start_epoch_ns(video_path)

    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame_index", "pts_s", "epoch_ns"])

        for i, pts_s in enumerate(frame_pts_list):
            epoch_ns = video_start_epoch_ns + round(pts_s * 1e9)
            writer.writerow([i, f"{pts_s:.9f}", epoch_ns])

    return video_start_epoch_ns


def main():
    parser = argparse.ArgumentParser(
        description="Extract video frames and frame timestamps in one step"
    )
    parser.add_argument("video_file", type=str, help="Path to MP4 video")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Optional frames output directory. Defaults to <video_stem>_frames",
    )
    parser.add_argument(
        "--timestamps-csv",
        type=str,
        default=None,
        help="Optional timestamps CSV path. Defaults to <video_stem>_frame_timestamps.csv",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["png", "jpg"],
        default="png",
        help="Output image format. Default: png",
    )
    args = parser.parse_args()

    video_path = Path(args.video_file)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found. Install with: sudo apt install ffmpeg")

    if shutil.which("ffprobe") is None:
        raise RuntimeError("ffprobe not found. Install with: sudo apt install ffmpeg")

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else video_path.with_name(f"{video_path.stem}_frames")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamps_csv = (
        Path(args.timestamps_csv)
        if args.timestamps_csv
        else video_path.with_name(f"{video_path.stem}_frame_timestamps.csv")
    )
    timestamps_csv.parent.mkdir(parents=True, exist_ok=True)

    frame_pts_list = get_frame_pts(video_path)
    video_start_epoch_ns = write_timestamps_csv(video_path, timestamps_csv, frame_pts_list)
    extracted_frames = extract_frames(video_path, output_dir, args.format)

    num_frames = len(extracted_frames)
    csv_rows = count_csv_rows(timestamps_csv)

    print("\n==============================")
    print(f"Video file           : {video_path}")
    print(f"Frames output dir    : {output_dir}")
    print(f"Frames extracted     : {num_frames}")
    print(f"Video start epoch ns : {video_start_epoch_ns}")
    print(f"Timestamps CSV       : {timestamps_csv}")
    print(f"Timestamp rows       : {csv_rows}")
    print("==============================")

    if csv_rows != num_frames:
        print(
            "Warning: frame count does not match timestamp CSV row count.\n"
            "You should fix this before writing the ROS bag."
        )
    else:
        print("Frame count matches timestamp CSV.")

    print("\nFirst few timestamp rows:")
    for i, pts_s in enumerate(frame_pts_list[:10]):
        epoch_ns = video_start_epoch_ns + round(pts_s * 1e9)
        print(f"{i}, {pts_s:.9f}, {epoch_ns}")

    print("\nFirst few extracted frame files:")
    for p in extracted_frames[:10]:
        print(p.name)


if __name__ == "__main__":
    main()
