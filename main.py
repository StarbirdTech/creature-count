import cv2
import numpy as np
import onnxruntime as ort
from getmodel import check_and_download_yolov8n_onnx

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
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and destroy all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
