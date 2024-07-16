import cv2
from ultralytics import YOLO

# Load the model
model = YOLO("dataaugm.pt")

# Define the video paths
video_path = r"C:\Users\cagla\OneDrive\Masaüstü\yoloproje\video.mp4"
output_path = r"C:\Users\cagla\OneDrive\Masaüstü\yoloproje\output_video.mp4"

# Open the video capture
cap = cv2.VideoCapture(video_path)

# Get video properties
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for output video
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the output video writer
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

while cap.isOpened():
    success, frame = cap.read()

    if success:
        results = model.track(frame, persist=True, show=False, tracker="botsort.yaml")

        annotated_frame = results[0].plot()

        # Get the bounding boxes, labels, and confidence scores from the results
        for result in results[0].boxes:
            x1, y1, x2, y2 = result.xyxy[0]
            cls = int(result.cls[0])  # Convert tensor to int
            conf = result.conf[0]  # Get confidence score
            label = f"{model.names[cls]} {conf:.2f}"  # Get class name from model.names

            # Draw the bounding box with a thinner line
            box_thickness = 1  # Adjust the thickness of the bounding box
            cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), box_thickness)

            # Draw the label with a smaller font
            font_scale = 0.5  # Adjust the font scale to make it smaller
            font_thickness = 1
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)

            # Draw the rectangle (label background) with thinner lines
            bg_thickness = -1  # Adjust the thickness of the background rectangle
            cv2.rectangle(annotated_frame, (int(x1), int(y1) - h - 10), (int(x1) + w, int(y1)), (0, 255, 0), bg_thickness)

            # Put the text on the rectangle
            cv2.putText(annotated_frame, label, (int(x1), int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), font_thickness)

        # Write the annotated frame to the output video
        out.write(annotated_frame)

        # Display the frame
        cv2.imshow("YOLOv8 Tracking", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()