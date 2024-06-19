import cv2
import math
import numpy as np
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO("yolov8n.pt")

# Open the video file
video_path = "C:/Users/micky/Documents/Ultra/bond_noise.mp4"
cap = cv2.VideoCapture(video_path)

# Get video properties
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
original_fps = cap.get(cv2.CAP_PROP_FPS)

# Create a VideoWriter object for the output video
output_path = "C:/Users/micky/Documents/Ultra/output_video_3.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output = cv2.VideoWriter(output_path, fourcc, original_fps, (width, height))

# Median filter kernel size
median_kernel_size = 3

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Apply median filter to remove salt and pepper noise
    denoised_frame = cv2.medianBlur(frame, median_kernel_size)

    # Perform object detection and tracking using YOLOv8
    results = model.track(denoised_frame, show=True)

    # Draw the bounding boxes and distance on the denoised frame
    for result in results:
        boxes = result.boxes
        for box in boxes:
            if box.cls == 0:  # 0 represents the 'person' class
                x1, y1, x2, y2 = [int(coord) for coord in box.xyxy[0]]
                center1 = ((x1 + x2) // 2, (y1 + y2) // 2)

                for other_box in boxes:
                    if other_box.cls == 0 and other_box.id != box.id:
                        x3, y3, x4, y4 = [int(coord) for coord in other_box.xyxy[0]]
                        center2 = ((x3 + x4) // 2, (y3 + y4) // 2)

                        # Calculate the Euclidean distance between the centers of the bounding boxes
                        distance = math.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)

                        # Draw the bounding boxes and distance on the denoised frame
                        cv2.rectangle(denoised_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.rectangle(denoised_frame, (x3, y3), (x4, y4), (0, 255, 0), 2)
                        text_bg_height = 30
                        cv2.rectangle(denoised_frame, (x1, y1 - text_bg_height), (x1 + 180, y1), (0, 0, 0), -1)
                        cv2.putText(denoised_frame, f"Distance: {int(distance)} pixels", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("YOLOv8 Object Detection", denoised_frame)
    output.write(denoised_frame)

    if cv2.waitKey(24) & 0xFF == ord('q'):
        break

cap.release()
output.release()
cv2.destroyAllWindows()
