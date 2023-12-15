import cv2
from ultralytics import YOLO
from ipSearch import get_ip

# Load the YOLOv8 model
model = YOLO("yolov8l-custom-birds-300.pt")

# Open the video file
video1 = "https://scontent.cdninstagram.com/v/t66.30100-16/321178736_1021873165551281_1779435619470033965_n.mp4?_nc_ht=scontent.cdninstagram.com&_nc_cat=105&_nc_ohc=YiX2MZK-lH4AX9_ckXg&edm=APs17CUBAAAA&ccb=7-5&oh=00_AfB_dhLkFrWGlGzL59pFyrVkWGCZamfjnfPD2tkc6Gek0A&oe=657D0085&_nc_sid=10d13b"
video2 = "https://scontent.cdninstagram.com/v/t66.30100-16/10000000_304046049161700_6126802921111712994_n.mp4?_nc_ht=scontent.cdninstagram.com&_nc_cat=100&_nc_ohc=mRYNbFl9eksAX8hmTbM&edm=APs17CUBAAAA&ccb=7-5&oh=00_AfDpFfAYkOssx4SEIpIAFjEczlDG6ywNmrEZV7CEV97YkQ&oe=657C81FC&_nc_sid=10d13b"
video3 = "https://scontent.cdninstagram.com/v/t50.2886-16/402149600_2636455536503812_6857444293256843050_n.mp4?_nc_ht=scontent.cdninstagram.com&_nc_cat=104&_nc_ohc=V3fYMYcqohkAX9b6QkJ&edm=APs17CUBAAAA&ccb=7-5&oh=00_AfD5J9YaX8AAhjdtX34RCyzrYCf-1g8O252LP5bfQUJ8vA&oe=657D051C&_nc_sid=10d13b"
video4 = "https://scontent.cdninstagram.com/v/t50.2886-16/404729560_259326877123898_6038490422009591577_n.mp4?_nc_ht=scontent.cdninstagram.com&_nc_cat=102&_nc_ohc=qiv5ddeo8bgAX_cgKjQ&edm=APs17CUBAAAA&ccb=7-5&oh=00_AfAg-Qjj1vfhpfOI96dDmHIeXJ8uIWzMA4VBxpKyqBocpw&oe=657CE4A2&_nc_sid=10d13b"

# target_device_name = "esp32-cam"
# ip_address = f"http://{get_ip(target_device_name)}:81/"

# print("Found {} at {}".format(target_device_name, ip_address))

cap = cv2.VideoCapture(video2)

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv8 inference on the frame
        results = model(frame)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Display the annotated frame
        cv2.imshow("YOLOv8 Inference", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
