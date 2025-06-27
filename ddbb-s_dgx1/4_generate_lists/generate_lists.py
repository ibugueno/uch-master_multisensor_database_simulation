# file: generate_lists.py

import os
import argparse
from pathlib import Path

def collect_image_paths(input_root: str) -> list:
    print(f"[DEBUG] Searching for .jpg files in {input_root}...")
    jpg_files = list(Path(input_root).rglob("*.jpg"))
    print(f"[DEBUG] Found {len(jpg_files)} .jpg files.")
    return jpg_files

def build_target_path(img_path: Path, replace_map: dict, ext: str = None) -> Path:
    p = str(img_path)
    for old, new in replace_map.items():
        p = p.replace(old, new)
    if ext:
        p = str(Path(p).with_suffix(ext))
    return Path(p)

def group_by_scene(paths: list) -> dict:
    grouped = {f"scene_{i}": [] for i in range(4)}
    for path in paths:
        parts = path.parts
        for i in range(4):
            scene_key = f"scene_{i}"
            if scene_key in parts:
                grouped[scene_key].append(path)
                break
    grouped = {k: v for k, v in grouped.items() if v}
    print(f"[DEBUG] Grouped into {len(grouped)} scene(s): {list(grouped.keys())}")
    return grouped

def process_sensor(sensor: str, base_root: str, output_dir: str, input_subpath: str, replace_map: dict):
    input_root = os.path.join(base_root, input_subpath)
    all_paths = collect_image_paths(input_root)

    if sensor == "asus":
        filtered_paths = [p for i, p in enumerate(sorted(all_paths)) if i % 33 == 0]
        print(f"[DEBUG] ASUS: Filtered to {len(filtered_paths)} images with index % 33 == 0")
    else:
        filtered_paths = all_paths

    scenes = group_by_scene(filtered_paths)

    for scene, img_paths in scenes.items():
        lines = {
            "data": [],
            "mask-seg": [],
            "det-bbox-abs": [],
            "pose6d-abs": []
        }

        print(f"[DEBUG] Processing sensor: {sensor}, scene: {scene} with {len(img_paths)} images")

        for p in img_paths:
            p = p.resolve()
            lines["data"].append(str(p))
            lines["mask-seg"].append(str(build_target_path(p, replace_map["mask-seg"])))
            lines["det-bbox-abs"].append(str(build_target_path(p, replace_map["det-bbox-abs"], ".txt")))
            lines["pose6d-abs"].append(str(build_target_path(p, replace_map["pose6d-abs"], ".txt")))

        for task, paths in lines.items():
            out_path = Path(output_dir) / f"{sensor}_{task}_{scene}.txt"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, 'w') as f:
                f.write('\n'.join(paths))
            print(f"[DEBUG] Saved: {out_path} with {len(paths)} entries")

def generate_txt_files(base_root: str, output_dir: str):
    config = {
        "davis346": {
            "input_subpath": "events-frames/davis346",
            "replace_map": {
                "mask-seg": {"events-frames": "frames", "events_noisy": "masks-seg"},
                "det-bbox-abs": {"events-frames": "frames", "events_noisy": "det-bbox-abs"},
                "pose6d-abs": {"events-frames": "frames", "events_noisy": "pose6d-abs"},
            }
        },
        "evk4": {
            "input_subpath": "events-frames/evk4",
            "replace_map": {
                "mask-seg": {"events-frames": "frames", "events_noisy": "masks-seg"},
                "det-bbox-abs": {"events-frames": "frames", "events_noisy": "det-bbox-abs"},
                "pose6d-abs": {"events-frames": "frames", "events_noisy": "pose6d-abs"},
            }
        },
        "asus": {
            "input_subpath": "frames/asus/images",
            "replace_map": {
                "mask-seg": {"images": "masks-seg"},
                "det-bbox-abs": {"images": "det-bbox-abs"},
                "pose6d-abs": {"images": "pose6d-abs"},
            }
        }
    }

    for sensor, cfg in config.items():
        process_sensor(sensor, base_root, output_dir, cfg["input_subpath"], cfg["replace_map"])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate dataset path .txt files for segmentation, detection, and pose tasks.")
    parser.add_argument('--base_root', required=True, help='Root path where events-frames and frames directories are located.')
    parser.add_argument('--output_dir', required=True, help='Directory to save the output .txt files.')
    args = parser.parse_args()

    generate_txt_files(args.base_root, args.output_dir)
