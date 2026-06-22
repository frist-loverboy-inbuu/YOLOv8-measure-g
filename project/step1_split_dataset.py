# ============================================================
# step1_split_dataset.py —— 自动分割数据集
# 
# 用途：把你标注好的图片和txt文件，自动按8:2分到train/val
#
# ▶ 使用前修改：
#   IMAGE_DIR  = 你存放所有球图片的文件夹
#   LABEL_DIR  = labelImg生成的.txt标注文件所在文件夹
#               （通常和图片在同一个文件夹）
#   OUTPUT_DIR = 输出到哪里（就是 ball.yaml 里的 path）
#
# ▶ 运行方式：python step1_split_dataset.py
# ============================================================

import os
import shutil
import random

# ──────────────────────────────────────────────
# ⚠️ 修改这里：你的图片和标注文件路径
IMAGE_DIR  = "dataset/images"            # 图片文件夹
LABEL_DIR  = "dataset/labels"            # 标注文件文件夹
OUTPUT_DIR = "dataset"                   # 输出目录
TRAIN_RATIO = 0.8                      # 80%训练，20%验证
# ──────────────────────────────────────────────

# 支持的图片格式
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

def main():
    # 建立输出文件夹结构
    for split in ["train", "val"]:
        os.makedirs(os.path.join(OUTPUT_DIR, "images", split), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_DIR, "labels", split), exist_ok=True)

    # 找出所有有对应标注文件的图片
    all_images = []
    for f in os.listdir(IMAGE_DIR):
        ext = os.path.splitext(f)[1].lower()
        if ext in IMG_EXTS:
            label_file = os.path.splitext(f)[0] + ".txt"
            label_path = os.path.join(LABEL_DIR, label_file)
            if os.path.exists(label_path):
                all_images.append(f)
            else:
                print(f"⚠️  跳过（没有标注文件）: {f}")

    if len(all_images) == 0:
        print("❌ 没找到任何图片+标注对，请检查路径")
        return

    # 随机打乱后分割
    random.shuffle(all_images)
    split_idx = int(len(all_images) * TRAIN_RATIO)
    train_list = all_images[:split_idx]
    val_list   = all_images[split_idx:]

    # 复制文件
    def copy_files(file_list, split_name):
        for img_name in file_list:
            label_name = os.path.splitext(img_name)[0] + ".txt"
            shutil.copy(
                os.path.join(IMAGE_DIR, img_name),
                os.path.join(OUTPUT_DIR, "images", split_name, img_name)
            )
            shutil.copy(
                os.path.join(LABEL_DIR, label_name),
                os.path.join(OUTPUT_DIR, "labels", split_name, label_name)
            )

    copy_files(train_list, "train")
    copy_files(val_list,   "val")

    print(f"\n✅ 数据集分割完成！")
    print(f"   训练集：{len(train_list)} 张")
    print(f"   验证集：{len(val_list)} 张")
    print(f"   输出目录：{OUTPUT_DIR}")

if __name__ == "__main__":
    main()
