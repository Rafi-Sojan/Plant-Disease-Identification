import os

DATA_DIR = "plantvillage_dataset"  

print("Sanity check running")

print(os.listdir(DATA_DIR))
print(os.listdir(os.path.join(DATA_DIR, "train"))[:5])
print(os.listdir(os.path.join(DATA_DIR, "val"))[:5])
print(os.listdir(os.path.join(DATA_DIR, "test"))[:5])
