import os
import shutil
from sklearn.model_selection import train_test_split

DATA_DIR = "data"
OUTPUT_DIR = "data_splits"

classes = ["with_mask", "without_mask"]
splits = ["train", "val", "test"]

for split in splits:
    for cls in classes:
        os.makedirs(os.path.join(OUTPUT_DIR, split, cls), exist_ok=True)

for cls in classes:
    class_dir = os.path.join(DATA_DIR, cls)
    images = [f for f in os.listdir(class_dir) if f.lower().endswith((".jpg", ".png", ".jpeg"))]

    train_imgs, temp_imgs = train_test_split(images, test_size=0.3, random_state=42)
    val_imgs, test_imgs = train_test_split(temp_imgs, test_size=0.5, random_state=42)

    for img_name, split in [
        *[(i, "train") for i in train_imgs],
        *[(i, "val") for i in val_imgs],
        *[(i, "test") for i in test_imgs],
    ]:
        src = os.path.join(class_dir, img_name)
        dst = os.path.join(OUTPUT_DIR, split, cls, img_name)
        shutil.copy2(src, dst)

print("Done splitting dataset!")
