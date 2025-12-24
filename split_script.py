import os
import shutil
import random

RAW_DIR = os.path.join("plantvillage_dataset", "plantvillage_dataset")
OUT_DIR = "plantvillage_dataset"

SPLITS = {
    "train": 0.7,
    "val": 0.15,
    "test": 0.15
}

IMG_EXTENSIONS = (".jpg", ".jpeg", ".png")

random.seed(42)


for split in SPLITS:
    os.makedirs(os.path.join(OUT_DIR, split), exist_ok=True)

for cls in os.listdir(RAW_DIR):
    cls_path = os.path.join(RAW_DIR, cls)

    #skip non-class folders
    if not os.path.isdir(cls_path):
        continue

    print(f"\nProcessing class: {cls}")

    #take image files
    images = [
        f for f in os.listdir(cls_path)
        if f.lower().endswith(IMG_EXTENSIONS)
        and os.path.isfile(os.path.join(cls_path, f))
    ]

    if len(images) == 0:
        print(f"Skipping {cls} (no images found)")
        continue

    random.shuffle(images)

    n = len(images)
    train_end = int(SPLITS["train"] * n)
    val_end = train_end + int(SPLITS["val"] * n)

    split_files = {
        "train": images[:train_end],
        "val": images[train_end:val_end],
        "test": images[val_end:]
    }

    for split, files in split_files.items():
        split_cls_dir = os.path.join(OUT_DIR, split, cls)
        os.makedirs(split_cls_dir, exist_ok=True)

        for file in files:
            src = os.path.join(cls_path, file)
            dst = os.path.join(split_cls_dir, file)
            shutil.copy2(src, dst)

    print(f"{cls}: {len(images)} images split")

print("\nData split complete")
