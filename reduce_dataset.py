# reduce_dataset.py
import json
import random
import shutil
from pathlib import Path
from collections import defaultdict
import pandas as pd

# ------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------
ROOT = Path(__file__).parent
DATA_ROOT = ROOT / "data" / "pklot"
SPLITS = ["train", "valid", "test"]
IMAGES_PER_SPLIT = 500          # <-- Change here if you want more/less
SEED = 42
random.seed(SEED)

# Output folders
REDUCED_ROOT = ROOT / "data" / "pklot_reduced"
# ------------------------------------------------------------------


def load_coco(split: str):
    json_path = DATA_ROOT / split / "_annotations.coco.json"
    with open(json_path) as f:
        return json.load(f)


def save_coco(data, split: str):
    out_dir = REDUCED_ROOT / split
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "_annotations.coco.json", "w") as f:
        json.dump(data, f, indent=2)


def copy_image(src_path: Path, dst_dir: Path):
    dst_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_path, dst_dir / src_path.name)


def reduce_split(split: str):
    print(f"\n--- Reducing {split.upper()} split ---")
    coco = load_coco(split)
    img_dict = {img["id"]: img for img in coco["images"]}

    # Group annotations by image_id
    ann_by_img = defaultdict(list)
    for ann in coco["annotations"]:
        ann_by_img[ann["image_id"]].append(ann)

    # Build list of (image_id, occupied_count, total_count)
    candidates = []
    for img_id, anns in ann_by_img.items():
        occupied = sum(1 for a in anns if a["category_id"] == 1)
        total = len(anns)
        candidates.append((img_id, occupied, total))

    # Sort by occupancy ratio to preserve balance
    candidates.sort(key=lambda x: x[1] / x[2] if x[2] > 0 else 0)

    # Pick 500 images evenly from low → high occupancy
    step = max(1, len(candidates) // IMAGES_PER_SPLIT)
    selected_img_ids = [c[0] for c in candidates[::step][:IMAGES_PER_SPLIT]]

    # If not enough, fill randomly
    if len(selected_img_ids) < IMAGES_PER_SPLIT:
        remaining = [c[0] for c in candidates if c[0] not in selected_img_ids]
        selected_img_ids += random.sample(remaining, IMAGES_PER_SPLIT - len(selected_img_ids))

    selected_img_ids = selected_img_ids[:IMAGES_PER_SPLIT]
    print(f"Selected {len(selected_img_ids)} images for {split}")

    # Build new COCO
    new_images = [img_dict[img_id] for img_id in selected_img_ids]
    new_ann_ids = set()
    new_annotations = []
    for img_id in selected_img_ids:
        for ann in ann_by_img[img_id]:
            if ann["id"] not in new_ann_ids:
                new_annotations.append(ann)
                new_ann_ids.add(ann["id"])

    new_coco = {
        "images": new_images,
        "annotations": new_annotations,
        "categories": coco["categories"]
    }

    # Save JSON
    save_coco(new_coco, split)

    # Copy images
    src_img_dir = DATA_ROOT / split
    dst_img_dir = REDUCED_ROOT / split
    for img in new_images:
        src_path = src_img_dir / img["file_name"]
        if src_path.exists():
            copy_image(src_path, dst_img_dir)
        else:
            print(f"Warning: Image not found: {src_path}")

    print(f"{split} reduced → {len(new_images)} images, {len(new_annotations)} annotations")


def main():
    print("PKLOT-640 REDUCTION TOOL")
    print(f"Target: {IMAGES_PER_SPLIT} images per split")
    print(f"Output: {REDUCED_ROOT}")

    for split in SPLITS:
        reduce_split(split)

    print("\nREDUCTION COMPLETE!")
    print(f"New dataset ready at: {REDUCED_ROOT}")
    print("Now update main.py to use pklot_reduced instead of pklot")


if __name__ == "__main__":
    main()