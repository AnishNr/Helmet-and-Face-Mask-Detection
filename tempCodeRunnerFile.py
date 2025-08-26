import cv2
import math
import cvzone
from ultralytics import YOLO

# Capture video from the default webcam
cap = cv2.VideoCapture(0)

# Load the YOLO model with custom weights
model = YOLO(r"weights/hemletYoloV8_100epochs.pt")

# Class names for helmet detection
classNames = ['Without Helmet', 'With Helmet']

# Check if webcam is opened successfully
if not cap.isOpened():
    print("Error: Cannot open webcam")
    exit()

while True:
    # Read frame from the webcam
    success, img = cap.read()

    # If frame is not captured correctly
    if not success:
        print("Failed to capture image from webcam")
        break

    # Run YOLO model on the captured frame
    results = model(img)

    # Process detection results
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Draw bounding box
            w, h = x2 - x1, y2 - y1
            cvzone.cornerRect(img, (x1, y1, w, h))

            # Get confidence and class
            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])

            # Check if the detected class index is valid
            if cls < len(classNames):
                # Display class name and confidence
                cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)), scale=1, thickness=1)
            else:
                # Debugging: Print if the class index is out of range
                print(f"Detected class ID {cls} is out of range!")

    # Show the frame with the detections
    cv2.imshow("Webcam", img)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release webcam and close windows
cap.release()
cv2.destroyAllWindows()
