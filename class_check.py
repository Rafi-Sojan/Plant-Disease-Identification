import os

base = "plantvillage_dataset"
for split in ["train", "val", "test"]:
    classes = sorted(os.listdir(os.path.join(base, split)))
    print(split, len(classes))
    print(classes)
