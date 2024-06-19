import cv2
import math
import numpy as np
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO("yolov8n.pt")

# Load the image
img_path = "C:/Users/micky/Documents/Ultra/screenshot.png"
img = cv2.imread(img_path)

# Detect objects in the image using YOLOv8
results = model(img)

# Extract the bounding boxes for the 'person' class
person_bboxes = results[0].boxes[results[0].boxes.cls == 0].xyxy.cpu().numpy()

# Calculate the distance between bounding boxes
for bbox1 in person_bboxes:
    x1, y1, x2, y2 = [int(coord) for coord in bbox1]
    center1 = ((x1 + x2) // 2, (y1 + y2) // 2)

    for bbox2 in person_bboxes:
        if np.array_equal(bbox1, bbox2):
            continue  # Skip the same bounding box

        x3, y3, x4, y4 = [int(coord) for coord in bbox2]
        center2 = ((x3 + x4) // 2, (y3 + y4) // 2)

        # Calculate the Euclidean distance between the centers of the bounding boxes
        distance = math.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)

        # Draw the bounding boxes and distance on the image
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.rectangle(img, (x3, y3), (x4, y4), (0, 255, 0), 2)

        # Create a black background for the text
        text_bg_height = 30
        cv2.rectangle(img, (x1, y1 - text_bg_height), (x1 + 180, y1), (0, 0, 0), -1)

        # Draw the distance text with a larger font and contrasting color
        cv2.putText(img, f"Distance: {int(distance)} pixels", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

# Display the image with bounding boxes and distances
cv2.imshow("YOLOv8 Object Detection", img)
cv2.waitKey(0)
cv2.destroyAllWindows()