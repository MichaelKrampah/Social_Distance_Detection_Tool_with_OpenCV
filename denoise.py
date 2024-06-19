import cv2
from ultralytics import YOLO

# Load your pretrained YOLOv8 model
model = YOLO("yolov8n.pt")

# Open the video filec
video_path = "C:/Users/micky/Documents/Ultra/bond_noise.mp4"
cap = cv2.VideoCapture(video_path)

# Get video properties
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
original_fps = cap.get(cv2.CAP_PROP_FPS)

# Create a VideoWriter object for the output video
output_path = "C:/Users/micky/Documents/Ultra/output_video_2.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output = cv2.VideoWriter(output_path, fourcc, original_fps, (width, height))

# Median filter kernel size
median_kernel_size = 3  # Adjust this value to control the amount of filtering

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Apply median filter to remove salt and pepper noise
    denoised_frame = cv2.medianBlur(frame, median_kernel_size)

    # Perform object detection using YOLOv8
    results = model(denoised_frame)
    person_bboxes = results[0].boxes[results[0].boxes.cls == 0].xyxy.cpu().numpy()

    for bbox in person_bboxes:
        x1, y1, x2, y2 = [int(coord) for coord in bbox]
        cv2.rectangle(denoised_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imshow("YOLOv8 Object Detection", denoised_frame)
    output.write(denoised_frame)

    if cv2.waitKey(24) & 0xFF == ord('q'):
        break

cap.release()
output.release()
cv2.destroyAllWindows()