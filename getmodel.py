import os
import subprocess

def check_and_download_yolov8n_onnx(model_dir='models', model_name='yolov8n.onnx'):
    model_path = os.path.join(model_dir, model_name)

    # Check if the model directory exists
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Check if the model file exists
    if not os.path.isfile(model_path):
        print(f"Model '{model_name}' not found in '{model_dir}'. Downloading and exporting...")

        # Download the pre-trained YOLOv8n model
        subprocess.run(['pip', 'install', 'ultralytics'])

        # Export the model to ONNX format
        subprocess.run([
            'yolo', 'export', 
            f'model={model_name.replace(".onnx", ".pt")}', 
            'imgsz=640', 'format=onnx', 'opset=12', 'simplify'
        ])

        # Move the model to the desired directory
        os.rename(model_name, model_path)

        print(f"Exported model to '{model_path}'")
    else:
        print(f"Model '{model_name}' found in '{model_dir}'.")

check_and_download_yolov8n_onnx()
