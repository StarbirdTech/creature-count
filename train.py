from ultralytics import YOLO
import torch

if not torch.cuda.is_available():
    temp = input("Running on CPU. Continue? (y/n) ").lower() == "y"
    if not temp:
        exit(0)

YOLO("yolov8n.pt").train(
    data="datasets/nabirds",
    imgsz=640,
    epochs=100,
    batch=8,
    name="yolov8l",
)
