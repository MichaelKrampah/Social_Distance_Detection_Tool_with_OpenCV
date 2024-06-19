import cv2
import math
import numpy as np
from ultralytics import YOLO
from scipy.spatial import distance as dist

# Estimated camera calibration parameters
camera_matrix = np.array([[1000.0, 0.0, 320.0],
                           [0.0, 1000.0, 240.0],
                           [0.0, 0.0, 1.0]])
dist_coeffs = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

# Pixel-to-metric conversion factor (adjust this value as needed)
pixel_to_meter_ratio = 0.0002645833  # 1 pixel = 0.01 meters (adjust this value based on your setup)

# Set threshold distance in meters
threshold_distance = 0.8

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
output_path = "C:/Users/micky/Documents/Ultra/final_output2.mp4"
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
        for i, box in enumerate(boxes):
            if box.cls == 0:  # 0 represents the 'person' class
                x1, y1, x2, y2 = [int(coord) for coord in box.xyxy[0]]
                centroid1 = ((x1 + x2) // 2, (y1 + y2) // 2)
                for other_box in boxes[i+1:]:
                    if other_box.cls == 0:
                        x3, y3, x4, y4 = [int(coord) for coord in other_box.xyxy[0]]
                        centroid2 = ((x3 + x4) // 2, (y3 + y4) // 2)
                        # Calculate the Euclidean distance between the centroids
                        distance_pixels = dist.euclidean(centroid1, centroid2)
                        distance_meters = distance_pixels * pixel_to_meter_ratio

                        # Set bounding box color based on threshold distance
                        if distance_meters < threshold_distance:
                            box_color1 = (0, 0, 255)  # Red
                            box_color2 = (0, 0, 255)  # Red
                        else:
                            box_color1 = (0, 255, 0)  # Green
                            box_color2 = (0, 255, 0)  # Green

                        # Draw the bounding boxes and distance on the denoised frame
                        cv2.rectangle(denoised_frame, (x1, y1), (x2, y2), box_color1, 2)
                        cv2.rectangle(denoised_frame, (x3, y3), (x4, y4), box_color2, 2)
                        text_bg_height = 30
                        cv2.rectangle(denoised_frame, (x1, y1 - text_bg_height), (x1 + 180, y1), (0, 0, 0), -1)
                        cv2.putText(denoised_frame, f"Distance: {distance_meters:.2f} m", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("YOLOv8 Object Detection", denoised_frame)
    output.write(denoised_frame)

    if cv2.waitKey(24) & 0xFF == ord('q'):
        break

cap.release()
output.release()
cv2.destroyAllWindows()
