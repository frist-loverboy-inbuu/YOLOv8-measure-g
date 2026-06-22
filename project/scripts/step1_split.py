import shutil
import random
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

def main(image_dir, label_dir, output_dir, train_ratio=0.8):
    image_dir = Path(image_dir)
    label_dir = Path(label_dir)
    output_dir = Path(output_dir)

    print(f"Image dir:  {image_dir}")
    print(f"Label dir:  {label_dir}")
    print(f"Output dir: {output_dir}")

    for split in ["train", "val"]:
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    all_images = []
    for f in image_dir.iterdir():
        ext = f.suffix.lower()
        if ext in IMG_EXTS:
            label_path = label_dir / (f.stem + ".txt")
            if label_path.exists():
                all_images.append(f.name)
            else:
                print(f"  Skip (no label): {f.name}")

    if not all_images:
        print("No image+label pairs found.")
        return

    random.shuffle(all_images)
    split_idx = int(len(all_images) * train_ratio)
    train_list = all_images[:split_idx]
    val_list = all_images[split_idx:]

    def copy_files(file_list, split_name):
        for img_name in file_list:
            label_name = Path(img_name).stem + ".txt"
            shutil.copy(image_dir / img_name, output_dir / "images" / split_name / img_name)
            shutil.copy(label_dir / label_name, output_dir / "labels" / split_name / label_name)

    copy_files(train_list, "train")
    copy_files(val_list, "val")

    print(f"\nDataset split: train={len(train_list)}, val={len(val_list)}")


if __name__ == "__main__":
    import yaml
    config_path = Path(__file__).parent.parent / "config.yaml"
    if config_path.exists():
        with open(config_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        ds = cfg.get("dataset", {})
        image_dir = ds.get("image_dir", "dataset/images")
        label_dir = ds.get("label_dir", "dataset/labels")
        output_dir = ds.get("output_dir", "dataset")
        train_ratio = ds.get("train_ratio", 0.8)
    else:
        image_dir = "dataset/images"
        label_dir = "dataset/labels"
        output_dir = "dataset"
        train_ratio = 0.8

    main(image_dir, label_dir, output_dir, train_ratio)
