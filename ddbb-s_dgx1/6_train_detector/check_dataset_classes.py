# file: check_dataset_classes_fast.py

import os
from collections import Counter
from train_det_fasterrcnn import load_paths_det, CLASS_MAPPING

def list_dataset_classes_fast(sensor, input_dir, scene):
    data_txt = os.path.join(input_dir, f"{sensor}_data_scene_{scene}.txt")
    bbox_txt = os.path.join(input_dir, f"{sensor}_det-bbox-abs-10ms_scene_{scene}.txt")
    image_paths, bbox_paths = load_paths_det(data_txt, bbox_txt)

    class_counter = Counter()
    missing_classes = set()

    for img_path in image_paths:
        obj_class = os.path.basename(os.path.dirname(os.path.dirname(img_path)))
        class_counter[obj_class] += 1

        if obj_class not in CLASS_MAPPING:
            missing_classes.add(obj_class)


    print("\n=== ✅ Dataset Classes Summary ===")
    for cls, count in class_counter.items():
        print(f"Class '{cls}': {count} samples")

    if missing_classes:
        print("\n⚠️ WARNING: Classes not in CLASS_MAPPING:")
        for cls in missing_classes:
            print(f" - {cls}")
    else:
        print("\n✅ All classes are correctly mapped in CLASS_MAPPING.")

if __name__ == "__main__":
    list_dataset_classes_fast(sensor="evk4", input_dir="/app/input/dataloader/", scene=0)
