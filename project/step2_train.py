# ============================================================
# step2_train.py —— 训练球体检测模型
#
# 用途：用你标注的图片，微调 YOLOv8n 让它认识你的球
#
# ▶ 使用前修改：YAML_PATH 改成你的 ball.yaml 路径
# ▶ 运行方式：python step2_train.py
# ▶ 预计时间：30~50张图片约5~10分钟
# ▶ 结果：runs/detect/ball_detector/weights/best.pt
# ============================================================

from ultralytics import YOLO

# ──────────────────────────────────────────────
# ⚠️ 修改这里
YAML_PATH ="ball.yaml"  # ball.yaml 的路径
EPOCHS    = 50                       # 训练轮数，图片少就用50，多可以用100
BATCH     = 8                        # 显存不够就改成4，够就改成16
# ──────────────────────────────────────────────

def main():
    # 从官方预训练权重开始微调（会自动下载，需要网络）
    model = YOLO("yolov8n.pt")

    print("🚀 开始训练...")
    results = model.train(
        data=YAML_PATH,
        epochs=EPOCHS,
        imgsz=640,
        batch=BATCH,
        patience=20,           # 20轮没提升就提前停止
        name="ball_detector",  # 结果保存在 runs/detect/ball_detector/
        verbose=True,
    )

    print("\n✅ 训练完成！")
    print("📦 最佳权重路径：runs/detect/ball_detector/weights/best.pt")

if __name__ == "__main__":
    main()
