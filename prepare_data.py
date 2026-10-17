
from pathlib import Path
from PIL import Image
from sklearn.model_selection import train_test_split
import shutil

SOURCE = Path("cats_and_dogs_filtered")
OUTPUT = Path("data")

CLASSES = ["cats", "dogs"]

def valid_images(folder):
    images = []

    for path in sorted(folder.iterdir()):
        if not path.is_file():
            continue

        try:
            with Image.open(path) as img:
                img.verify()
            images.append(path)

        except Exception:
            print("Corrupted image:", path)

    return images


def copy_images(files, split, label):
    destination = OUTPUT / split / label
    destination.mkdir(parents=True, exist_ok=True)

    for path in files:
        shutil.copy2(path, destination / path.name)


if OUTPUT.exists():
    shutil.rmtree(OUTPUT)

for label in CLASSES:

    train_files = valid_images(
        SOURCE / "train" / label
    )

    remaining = valid_images(
        SOURCE / "validation" / label
    )

    val_files, test_files = train_test_split(
        remaining,
        test_size=0.5,
        random_state=42
    )

    copy_images(train_files, "train", label)
    copy_images(val_files, "val", label)
    copy_images(test_files, "test", label)

    print(
        label,
        "Train:", len(train_files),
        "Validation:", len(val_files),
        "Test:", len(test_files)
    )

print("Dataset preparation completed!")
