import cv2
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO("yolov8n.pt")

# Open the video file
video_path = "C:/Users/micky/Documents/Ultra/bond_noise.mp4"
cap = cv2.VideoCapture(video_path)

# Median filter kernel size
median_kernel_size = 3

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Apply median filter to remove salt and pepper noise
    denoised_frame = cv2.medianBlur(frame, median_kernel_size)

    # Perform object detection using YOLOv8
    results = model(denoised_frame)

    # Draw the bounding boxes and center coordinates on the denoised frame
    for result in results:
        boxes = result.boxes
        for box in boxes:
            if box.cls == 0:  # 0 represents the 'person' class
                x1, y1, x2, y2 = [int(coord) for coord in box.xyxy[0]]

                # Calculate the center coordinates of the bounding box
                box_center_x = (x1 + x2) // 2
                box_center_y = (y1 + y2) // 2

                # Draw the bounding box and center coordinates on the denoised frame
                cv2.rectangle(denoised_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(denoised_frame, (box_center_x, box_center_y), 5, (0, 0, 255), -1)
                cv2.putText(denoised_frame, f"X: {box_center_x}, Y: {box_center_y}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    cv2.imshow("YOLOv8 Object Detection", denoised_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()