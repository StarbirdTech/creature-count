from ultralytics import YOLO
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

base_model = "yolov8l.pt"
model = YOLO(base_model).to(device)

if __name__ == "__main__":
    # Training.
    model.train(
        data="dataset.yaml",
        device=device,
        imgsz=640,
        epochs=300,
        batch=8,
        name="yolov8_bird_species_big",
    )
