from ultralytics import YOLO

YAML_PATH = "ball.yaml"
EPOCHS = 50
BATCH = 8


def main():
    model = YOLO("yolov8n.pt")
    print("Starting training...")
    model.train(
        data=YAML_PATH,
        epochs=EPOCHS,
        imgsz=640,
        batch=BATCH,
        patience=20,
        name="ball_detector",
        verbose=True,
    )
    print("\nTraining done.")
    print("Best weights: runs/detect/ball_detector/weights/best.pt")


if __name__ == "__main__":
    main()
