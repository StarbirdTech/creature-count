import cv2
from ultralytics import YOLO

# Load the YOLOv8 model (you can replace "yolov8n.pt" with another model variant)
model = YOLO("yolov8n.pt")

# Start capturing video from the webcam
cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    raise IOError("Cannot open webcam")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Perform detection
    results = model(frame)

    # Render the results on the frame
    frame = results.render()[0]

    # Display the frame
    cv2.imshow('YOLOv8 Webcam', frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the VideoCapture object and close windows
cap.release()
cv2.destroyAllWindows()
