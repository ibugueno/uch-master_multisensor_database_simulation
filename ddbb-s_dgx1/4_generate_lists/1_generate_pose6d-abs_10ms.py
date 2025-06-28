#!/usr/bin/env python3

import os
from pathlib import Path
import csv
import shutil

def process_pose6d_and_bbox(pose6d_file, bbox_file, dst_file):
    # Leer pose6d (profundidad y quaterniones)
    with open(pose6d_file) as f:
        reader = csv.reader(f)
        header_pose = next(reader)
        row_pose = next(reader)
        # Asumimos estructura [depth_cm, qx, qy, qz, qw]
        depth_cm = float(row_pose[0])
        qx, qy, qz, qw = map(float, row_pose[1:5])

    # Leer bbox (xmin, ymin, xmax, ymax)
    with open(bbox_file) as f:
        reader = csv.reader(f)
        header_bbox = next(reader)
        row_bbox = next(reader)
        xmin, ymin, xmax, ymax = map(int, row_bbox)

    # Combinar
    combined_row = [xmin, ymin, xmax, ymax, depth_cm, qx, qy, qz, qw]

    # Debug info
    print(f"[INFO] Writing combined file to: {dst_file}")
    if WRITE_FILES:
        dst_file.parent.mkdir(parents=True, exist_ok=True)
        with open(dst_file, "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["xmin", "ymin", "xmax", "ymax", "depth_cm", "qx", "qy", "qz", "qw"])
            writer.writerow(combined_row)

def process_sensor(sensor):
    pose6d_abs_path = BASE_PATH / sensor / "pose6d-abs"
    bbox_abs_10ms_path = BASE_PATH / sensor / "det-bbox-abs-10ms"
    output_path = BASE_PATH / sensor / OUTPUT_DIRNAME

    print(f"[INFO] Processing sensor: {sensor}")
    for pose6d_file in pose6d_abs_path.rglob("*.txt"):
        relative = pose6d_file.relative_to(pose6d_abs_path)
        bbox_file = bbox_abs_10ms_path / relative
        dst_file = output_path / relative

        if bbox_file.exists():
            process_pose6d_and_bbox(pose6d_file, bbox_file, dst_file)
        else:
            print(f"[WARNING] Missing bbox file: {bbox_file}")

def main():
    for sensor in SENSORS:
        process_sensor(sensor)

if __name__ == "__main__":

    # Configurable flag
    WRITE_FILES = True

    BASE_PATH = Path("/app/input/frames")
    OUTPUT_DIRNAME = "pose6d-abs-10ms"
    SENSORS = ["asus", "evk4", "davis346"]

    main()
