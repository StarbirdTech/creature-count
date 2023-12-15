import cv2
import numpy as np
import onnxruntime as ort

import os
import subprocess


def check_and_download_yolov8n_onnx(model_dir="models", model_name="yolov8n.onnx"):
    model_path = os.path.join(model_dir, model_name)

    # Check if the model directory exists
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Check if the model file exists
    if not os.path.isfile(model_path):
        print(
            f"Model '{model_name}' not found in '{model_dir}'. Downloading and exporting..."
        )

        # Download the pre-trained YOLOv8n model
        subprocess.run(["pip", "install", "ultralytics"])

        # Export the model to ONNX format
        subprocess.run(
            [
                "yolo",
                "export",
                f'model={model_name.replace(".onnx", ".pt")}',
                "imgsz=640",
                "format=onnx",
                "opset=12",
                "simplify",
            ]
        )

        # Move the model to the desired directory
        os.rename(model_name, model_path)

        print(f"Exported model to '{model_path}'")
    else:
        print(f"Model '{model_name}' found in '{model_dir}'.")


def preprocess(image):
    # Resize and normalize the image
    image = cv2.resize(image, (640, 640))
    image = image / 255.0
    image = image.astype(np.float32)
    image = np.transpose(image, (2, 0, 1))
    image = np.expand_dims(image, axis=0)
    return image


def postprocess(outputs, frame):
    # Define your postprocessing steps here
    # This function should convert raw model outputs to bounding boxes and labels
    pass


def main():
    check_and_download_yolov8n_onnx()
    # Load the ONNX model
    session = ort.InferenceSession("yolov8n.onnx")
    input_name = session.get_inputs()[0].name

    # Open the webcam
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess the frame
        input_tensor = preprocess(frame)

        # Run inference
        outputs = session.run(None, {input_name: input_tensor})

        # Postprocessing to extract bounding boxes and draw them
        postprocess(outputs, frame)

        # Display the frame
        cv2.imshow("YOLOv8 Webcam", frame)

        # Exit on pressing the 'q' key
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Release the webcam and destroy all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
