import cv2
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO("yolov8l-custom-birds-300.pt")

# List of demo video files
videos = ["demo1.mp4", "demo2.mp4", "demo3.mp4", "demo4.mp4"]

for video in videos:
    cap = cv2.VideoCapture(video)
    # Define the codec and create VideoWriter object
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    output_name = video.split(".")[0] + "-render.mp4"
    out = cv2.VideoWriter(
        output_name, cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame_width, frame_height)
    )

    while cap.isOpened():
        success, frame = cap.read()
        if success:
            # Run YOLOv8 inference on the frame
            results = model.predict(frame, classes=[24, 41])
            # Visualize the results on the frame
            annotated_frame = results[0].plot()

            # Write the frame into the file 'output_name'
            out.write(annotated_frame)

            # Display the annotated frame
            cv2.imshow("YOLOv8 Inference", annotated_frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            break

    # Release the video capture and video write objects
    cap.release()
    out.release()

cv2.destroyAllWindows()
