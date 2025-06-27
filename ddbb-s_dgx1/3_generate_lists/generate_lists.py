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

def generate_txt_files_from_davis346(base_root: str, output_dir: str):
    davis_input_root = os.path.join(base_root, "events-frames/davis346")
    davis_paths = collect_image_paths(davis_input_root)
    scenes = group_by_scene(davis_paths)

    sensors = ["davis346", "evk4", "asus"]
    replacements = {
        "davis346": {
            "evk4": [
                ("davis346", "evk4")
            ],
            "asus": [
                ("events-frames/davis346/events_noisy", "frames/asus/images"),
                ("events-frames/davis346", "frames/asus/images"),
                ("events-frames", "frames"),
                ("events_noisy", "images")
            ]
        }
    }

    for scene, img_paths in scenes.items():
        lines_davis = {
            "data": [],
            "mask-seg": [],
            "det-bbox-abs": [],
            "pose6d-abs": []
        }

        print(f"[DEBUG] Processing scene: {scene} with {len(img_paths)} images")

        for p in img_paths:
            p = p.resolve()
            lines_davis["data"].append(str(p))
            lines_davis["mask-seg"].append(str(build_target_path(p, {"events-frames": "frames", "events_noisy": "masks-seg"})))
            lines_davis["det-bbox-abs"].append(str(build_target_path(p, {"events-frames": "frames", "events_noisy": "det-bbox-abs"}, ".txt")))
            lines_davis["pose6d-abs"].append(str(build_target_path(p, {"events-frames": "frames", "events_noisy": "pose6d-abs"}, ".txt")))

        # Guardar archivos davis346
        for task, paths in lines_davis.items():
            out_path = Path(output_dir) / f"davis346_{task}_{scene}.txt"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, 'w') as f:
                f.write('\n'.join(map(str, paths)))
            print(f"[DEBUG] Saved: {out_path} with {len(paths)} entries")

        # Reutilizar para evk4 y asus
        for sensor in ["evk4", "asus"]:
            for task in ["data", "mask-seg", "det-bbox-abs", "pose6d-abs"]:
                lines = []
                for path in lines_davis[task]:
                    new_path = path
                    for old, new in replacements["davis346"][sensor]:
                        new_path = new_path.replace(old, new)
                    if sensor == "asus":
                        if task == "det-bbox-abs" or task == "pose6d-abs":
                            new_path = new_path.replace(".jpg", ".txt")
                    lines.append(new_path)
                out_path = Path(output_dir) / f"{sensor}_{task}_{scene}.txt"
                with open(out_path, 'w') as f:
                    f.write('\n'.join(lines))
                print(f"[DEBUG] Saved: {out_path} with {len(lines)} entries")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate dataset path .txt files for segmentation, detection, and pose tasks.")
    parser.add_argument('--base_root', required=True, help='Root path where events-frames and frames directories are located.')
    parser.add_argument('--output_dir', required=True, help='Directory to save the output .txt files.')
    args = parser.parse_args()

    generate_txt_files_from_davis346(args.base_root, args.output_dir)
