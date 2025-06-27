#!/usr/bin/env python3

import os
from pathlib import Path
import csv
import shutil

def process_bbox_file(src_file, next_file, dst_file):
    with open(src_file) as f:
        reader = csv.reader(f)
        header = next(reader)
        src_row = next(reader)
        src_xmin, src_ymin, src_xmax, src_ymax = map(float, src_row)

    if next_file:
        with open(next_file) as f:
            reader = csv.reader(f)
            next_header = next(reader)
            next_row = next(reader)
        next_xmin, next_ymin, next_xmax, next_ymax = map(float, next_row)
        next_exists = True

        # Combine values
        xmin = min(src_xmin, next_xmin)
        ymin = min(src_ymin, next_ymin)
        xmax = max(src_xmax, next_xmax)
        ymax = max(src_ymax, next_ymax)
    else:
        xmin, ymin, xmax, ymax = src_xmin, src_ymin, src_xmax, src_ymax
        next_exists = False

    # Print debug info
    print(f"[INFO] Processing src: {src_file}")
    print(f"       Using next: {next_file if next_file else 'None'} (exists: {next_exists})")
    print(f"       Result -> xmin: {xmin}, ymin: {ymin}, xmax: {xmax}, ymax: {ymax}")
    print(f"       Writing to: {dst_file}" if WRITE_FILES else f"       [DRY RUN] Would write to: {dst_file}")

    if WRITE_FILES:
        dst_file.parent.mkdir(parents=True, exist_ok=True)
        with open(dst_file, "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerow([int(xmin), int(ymin), int(xmax), int(ymax)])

def process_orientation_folder(orientation_path, output_orientation_path):
    files = sorted(orientation_path.glob("image_*.txt"))
    indices = [int(f.stem.split("_")[1]) for f in files]
    index_to_file = dict(zip(indices, files))

    for idx in indices:
        src_file = index_to_file[idx]
        target_idx = idx + 10

        # Find target file
        if target_idx in index_to_file:
            next_file = index_to_file[target_idx]
        else:
            # Find max index <= target_idx within next 10
            next_indices = [i for i in indices if i > idx and i <= idx + 10]
            next_file = index_to_file[next_indices[-1]] if next_indices else None

        dst_file = output_orientation_path / src_file.name
        process_bbox_file(src_file, next_file, dst_file)

def copy_asus_det_bbox_abs(sensor):
    src = BASE_PATH / sensor / "det-bbox-abs"
    dst = BASE_PATH / sensor / OUTPUT_DIRNAME
    print(f"[INFO] Copying {src} -> {dst}")
    if WRITE_FILES:
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)

def main():
    for sensor in SENSORS:
        if sensor == "asus":
            copy_asus_det_bbox_abs(sensor)
            continue

        det_bbox_abs_path = BASE_PATH / sensor / "det-bbox-abs"
        output_base = BASE_PATH / sensor / OUTPUT_DIRNAME

        for scene_dir in det_bbox_abs_path.glob("scene_*"):
            for lum_dir in scene_dir.glob("*"):
                for obj_dir in lum_dir.glob("*"):
                    for orientation_dir in obj_dir.glob("orientation_*"):
                        relative_path = orientation_dir.relative_to(det_bbox_abs_path)
                        output_orientation_path = output_base / relative_path

                        print(f"[INFO] Processing orientation folder: {orientation_dir}")
                        process_orientation_folder(orientation_dir, output_orientation_path)

                        #break
                    #break
                #break
            #break


if __name__ == "__main__":

    # Configurable flag
    WRITE_FILES = True

    BASE_PATH = Path("/app/input/frames")
    OUTPUT_DIRNAME = "det-bbox-abs-10ms"
    SENSORS = ["asus", "evk4", "davis346"]

    main()
