import cv2
from ultralytics import YOLO

# Load your pretrained YOLOv8 model
model = YOLO("yolov8n.pt")

# Open the video file
video_path = "C:/Users/micky/Documents/Ultra/bond_noise.mp4"
cap = cv2.VideoCapture(video_path)

# Get video properties
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
original_fps = cap.get(cv2.CAP_PROP_FPS)  # Original frame rate

# Create a VideoWriter object for the output video
output_path = "C:/Users/micky/Documents/Ultra_Videos/output_video.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output = cv2.VideoWriter(output_path, fourcc, 1, (width, height))  # Decreased fps (slow motion)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Perform object detection using YOLOv8
    results = model(frame)
    person_bboxes = results[0].boxes[results[0].boxes.cls == 0].xyxy.cpu().numpy()

    for bbox in person_bboxes:
        x1, y1, x2, y2 = [int(coord) for coord in bbox]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imshow("YOLOv8 Object Detection", frame)
    output.write(frame)  # Write frame to output video

    if cv2.waitKey(24) & 0xFF == ord('q'):
        break

# Release video capture and close the window
cap.release()
output.release()
cv2.destroyAllWindows()
