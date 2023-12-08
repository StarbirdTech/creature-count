from ultralytics import YOLO
import torch
from dotenv import load_dotenv
import os

load_dotenv()

from roboflow import Roboflow

rf = Roboflow(api_key=os.getenv("MYLES_ROBOFLOW_API_KEY"))
project = rf.workspace("ecosorter-drvgm").project("plastic-bag-dataset")
dataset = project.version(1).download(
    "yolov8", location="./datasets/plastic-bag-dataset"
)

if not torch.cuda.is_available():
    if not input("Running on CPU. Continue? (y/n) ").lower() == "y":
        exit(0)

YOLO("yolov8n").train(
    data="./datasets/plastic-bag-dataset/data.yaml",  # Assuming this is the correct path
    epochs=100,
    imgsz=640,
    name="yolov8n-plastic-bag-dataset",
)
