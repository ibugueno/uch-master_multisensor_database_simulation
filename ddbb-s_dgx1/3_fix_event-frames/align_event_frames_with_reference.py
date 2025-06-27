# file: fix_event_frame_indices.py

import os
from pathlib import Path
import shutil
import re

def get_start_index(frame_dir: Path) -> int:
    jpgs = sorted(frame_dir.glob("image_*.jpg"))
    if not jpgs:
        return None
    match = re.search(r"image_(\d+)\.jpg", jpgs[0].name)
    return int(match.group(1)) if match else None

def fix_event_frames(events_root: Path, frames_root: Path, output_root: Path):
    orientation_dirs = list(events_root.rglob("orientation_*"))
    for ori_dir in orientation_dirs:
        rel_path = ori_dir.relative_to(events_root)
        frames_ori_dir = frames_root / rel_path

        if not frames_ori_dir.exists():
            print(f"[SKIP] No matching frames dir: {frames_ori_dir}")
            continue

        start_idx = get_start_index(frames_ori_dir)
        if start_idx is None:
            print(f"[SKIP] No image files in: {frames_ori_dir}")
            continue

        output_ori_dir = output_root / rel_path
        output_ori_dir.mkdir(parents=True, exist_ok=True)

        jpgs = sorted(ori_dir.glob("image_*.jpg"))
        for i, jpg_path in enumerate(jpgs):
            match = re.search(r"image_(\d+)\.jpg", jpg_path.name)
            if not match:
                continue
            offset = int(match.group(1))
            new_index = start_idx + offset
            new_name = f"image_{new_index:04d}.jpg"
            dst_path = output_ori_dir / new_name
            shutil.copy2(jpg_path, dst_path)

        print(f"[OK] Processed: {ori_dir} -> {output_ori_dir}")

if __name__ == "__main__":
    base_path = Path("../../input")
    for sensor in ["davis346", "evk4"]:
        events_dir = base_path / f"events-frames/{sensor}/events_noisy"
        frames_dir = base_path / f"frames/{sensor}/images"
        output_dir = base_path / f"events-frames2/{sensor}/events_noisy"
        fix_event_frames(events_dir, frames_dir, output_dir)
