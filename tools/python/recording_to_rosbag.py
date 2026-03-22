#!/usr/bin/env python3

import argparse
import csv
from pathlib import Path

import cv2
import pandas as pd

import rosbag2_py
from rclpy.serialization import serialize_message

from builtin_interfaces.msg import Time
from sensor_msgs.msg import CameraInfo, Image, Imu, MagneticField


# =========================
# Helpers
# =========================

def ns_to_time_msg(t_ns: int) -> Time:
    msg = Time()
    msg.sec = int(t_ns // 1_000_000_000)
    msg.nanosec = int(t_ns % 1_000_000_000)
    return msg


def load_sensor_csv(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    df = pd.read_csv(csv_path)

    required = ["x", "y", "z"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"{csv_path.name} missing columns {missing}. Found: {list(df.columns)}"
        )

    if "time" in df.columns:
        time_col = "time"
    elif "seconds_elapsed" in df.columns:
        raise ValueError(
            f"{csv_path.name} has seconds_elapsed but not absolute 'time'. "
            "This script needs absolute nanosecond timestamps for rosbag writing."
        )
    else:
        raise ValueError(
            f"{csv_path.name} missing time column. Found: {list(df.columns)}"
        )

    for c in [time_col, "x", "y", "z"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=[time_col, "x", "y", "z"]).reset_index(drop=True)
    if df.empty:
        raise ValueError(f"{csv_path.name} has no valid rows after numeric cleanup")

    df[time_col] = df[time_col].astype("int64")
    out = df[[time_col, "x", "y", "z"]].copy()
    out = out.rename(columns={time_col: "time"})
    return out


def load_frame_timestamps(csv_path: Path):
    rows = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        required = {"frame_index", "epoch_ns"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(
                f"{csv_path.name} missing required columns {sorted(required)}. "
                f"Found: {reader.fieldnames}"
            )
        for r in reader:
            rows.append({
                "frame_index": int(r["frame_index"]),
                "epoch_ns": int(r["epoch_ns"]),
            })
    if not rows:
        raise ValueError(f"No frame timestamp rows found in {csv_path}")
    return rows


def find_camera_assets(camera_dir: Path):
    mp4_files = sorted(camera_dir.glob("*.mp4"))
    if not mp4_files:
        raise RuntimeError(f"No MP4 found in {camera_dir}")

    video = mp4_files[0]
    stem = video.stem
    frames_dir = camera_dir / f"{stem}_frames"
    timestamps_csv = camera_dir / f"{stem}_frame_timestamps.csv"

    if not frames_dir.exists():
        raise RuntimeError(f"Frames dir not found: {frames_dir}")
    if not timestamps_csv.exists():
        raise RuntimeError(f"Timestamps CSV not found: {timestamps_csv}")

    return video, frames_dir, timestamps_csv


def make_camera_info_msg(width: int, height: int, stamp_ns: int, frame_id: str, args) -> CameraInfo:
    msg = CameraInfo()
    msg.header.stamp = ns_to_time_msg(stamp_ns)
    msg.header.frame_id = frame_id

    msg.width = width
    msg.height = height
    msg.distortion_model = "plumb_bob"
    msg.d = [args.k1, args.k2, args.p1, args.p2, args.k3]

    msg.k = [
        args.fx, 0.0, args.cx,
        0.0, args.fy, args.cy,
        0.0, 0.0, 1.0,
    ]

    msg.r = [
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0,
    ]

    msg.p = [
        args.fx, 0.0, args.cx, 0.0,
        0.0, args.fy, args.cy, 0.0,
        0.0, 0.0, 1.0, 0.0,
    ]

    return msg


# =========================
# ROS Writer
# =========================

def make_writer(output_bag: Path):
    writer = rosbag2_py.SequentialWriter()

    storage_options = rosbag2_py.StorageOptions(
        uri=str(output_bag),
        storage_id="mcap",
    )

    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format="cdr",
        output_serialization_format="cdr",
    )

    writer.open(storage_options, converter_options)

    topics = [
        ("/camera/image_raw", "sensor_msgs/msg/Image"),
        ("/camera/camera_info", "sensor_msgs/msg/CameraInfo"),
        ("/imu/data_raw", "sensor_msgs/msg/Imu"),
        ("/imu/mag", "sensor_msgs/msg/MagneticField"),
    ]

    for topic_id, (name, topic_type) in enumerate(topics):
        topic_metadata = rosbag2_py.TopicMetadata(
            id=topic_id,
            name=name,
            type=topic_type,
            serialization_format="cdr",
            offered_qos_profiles=[],
        )
        writer.create_topic(topic_metadata)

    return writer


# =========================
# Writers
# =========================

def write_camera(writer, frames_dir: Path, timestamps_csv: Path, args):
    rows = load_frame_timestamps(timestamps_csv)
    frames = sorted(frames_dir.glob("frame_*.*"))

    if len(rows) != len(frames):
        raise RuntimeError(
            f"Frame count mismatch: {len(frames)} images vs {len(rows)} timestamps"
        )

    for row, img_path in zip(rows, frames):
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"Failed to read image: {img_path}")

        stamp_ns = row["epoch_ns"]
        frame_id = args.camera_frame_id
        height, width = img.shape[:2]

        img_msg = Image()
        img_msg.header.stamp = ns_to_time_msg(stamp_ns)
        img_msg.header.frame_id = frame_id
        img_msg.height = height
        img_msg.width = width
        img_msg.encoding = "bgr8"
        img_msg.is_bigendian = 0
        img_msg.step = width * 3
        img_msg.data = img.tobytes()

        cam_info_msg = make_camera_info_msg(width, height, stamp_ns, frame_id, args)

        writer.write("/camera/image_raw", serialize_message(img_msg), stamp_ns)
        writer.write("/camera/camera_info", serialize_message(cam_info_msg), stamp_ns)


def write_imu(writer, gyro_csv: Path, accel_csv: Path, args):
    gyro = load_sensor_csv(gyro_csv)
    accel = load_sensor_csv(accel_csv)

    accel_times = accel["time"]

    for _, g in gyro.iterrows():
        t = int(g["time"])

        idx = (accel_times - t).abs().idxmin()
        a = accel.loc[idx]

        msg = Imu()
        msg.header.stamp = ns_to_time_msg(t)
        msg.header.frame_id = args.imu_frame_id

        msg.angular_velocity.x = float(g["x"])
        msg.angular_velocity.y = float(g["y"])
        msg.angular_velocity.z = float(g["z"])

        msg.linear_acceleration.x = float(a["x"])
        msg.linear_acceleration.y = float(a["y"])
        msg.linear_acceleration.z = float(a["z"])

        msg.orientation_covariance[0] = -1.0
        msg.angular_velocity_covariance = [args.gyro_var, 0.0, 0.0,
                                           0.0, args.gyro_var, 0.0,
                                           0.0, 0.0, args.gyro_var]
        msg.linear_acceleration_covariance = [args.accel_var, 0.0, 0.0,
                                              0.0, args.accel_var, 0.0,
                                              0.0, 0.0, args.accel_var]

        writer.write("/imu/data_raw", serialize_message(msg), t)


def write_mag(writer, mag_csv: Path, args):
    mag = load_sensor_csv(mag_csv)

    for _, row in mag.iterrows():
        t = int(row["time"])

        msg = MagneticField()
        msg.header.stamp = ns_to_time_msg(t)
        msg.header.frame_id = args.imu_frame_id

        msg.magnetic_field.x = float(row["x"])
        msg.magnetic_field.y = float(row["y"])
        msg.magnetic_field.z = float(row["z"])

        msg.magnetic_field_covariance = [args.mag_var, 0.0, 0.0,
                                         0.0, args.mag_var, 0.0,
                                         0.0, 0.0, args.mag_var]

        writer.write("/imu/mag", serialize_message(msg), t)


# =========================
# Main
# =========================

def main():
    parser = argparse.ArgumentParser(
        description="Convert one phone recording folder into a rosbag2/mcap with camera, camera_info, IMU, and optional magnetometer"
    )
    parser.add_argument("--recording-dir", required=True, type=str)
    parser.add_argument("--output-bag", required=True, type=str)

    parser.add_argument("--camera-frame-id", default="camera", type=str)
    parser.add_argument("--imu-frame-id", default="imu", type=str)

    # Camera calibration defaults from the user's calibration result
    parser.add_argument("--fx", type=float, default=1351.26516)
    parser.add_argument("--fy", type=float, default=1352.29639)
    parser.add_argument("--cx", type=float, default=503.750565)
    parser.add_argument("--cy", type=float, default=881.210066)
    parser.add_argument("--k1", type=float, default=0.26715002)
    parser.add_argument("--k2", type=float, default=0.10367042)
    parser.add_argument("--p1", type=float, default=-0.03004398)
    parser.add_argument("--p2", type=float, default=-0.01041310)
    parser.add_argument("--k3", type=float, default=-5.34811256)

    # Simple diagonal covariances
    parser.add_argument("--accel-var", type=float, default=0.04)
    parser.add_argument("--gyro-var", type=float, default=0.0004)
    parser.add_argument("--mag-var", type=float, default=0.01)

    args = parser.parse_args()

    rec = Path(args.recording_dir)
    if not rec.exists():
        raise FileNotFoundError(f"Recording dir not found: {rec}")

    gyro_csv = rec / "Gyroscope.csv"
    accel_csv = rec / "Accelerometer.csv"
    mag_csv = rec / "Magnetometer.csv"
    camera_dir = rec / "Camera"

    video, frames_dir, timestamps_csv = find_camera_assets(camera_dir)

    print("Detected:")
    print(" Video        :", video.name)
    print(" Frames dir   :", frames_dir.name)
    print(" Timestamp CSV:", timestamps_csv.name)
    print(" Accel CSV    :", accel_csv.name)
    print(" Gyro CSV     :", gyro_csv.name)
    print(" Magnetometer :", mag_csv.exists())

    writer = make_writer(Path(args.output_bag))

    print("Writing camera + camera_info...")
    write_camera(writer, frames_dir, timestamps_csv, args)

    print("Writing IMU...")
    write_imu(writer, gyro_csv, accel_csv, args)

    if mag_csv.exists():
        print("Writing magnetometer...")
        write_mag(writer, mag_csv, args)
    else:
        print("Skipping magnetometer: Magnetometer.csv not found")

    print("Done:", args.output_bag)


if __name__ == "__main__":
    main()
