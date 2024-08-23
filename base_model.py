import cv2
from ultralytics import YOLO

model = YOLO("dataaugm.pt")
video_path = r"C:\Users\cagla\OneDrive\Masaüstü\yoloproje\video.mp4"
cap = cv2.VideoCapture(video_path)

output_path = r"C:\Users\cagla\OneDrive\Masaüstü\yoloproje\output_video.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can also use 'XVID', 'X264', etc.
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

while cap.isOpened(): 
    success, frame = cap.read()

    if success:
        results = model.track(frame, persist=True, show=False, tracker="botsort.yaml")

        annotated_frame = results[0].plot()

        out.write( frame)

        cv2.imshow("YOLOv8 Tracking", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

cap.release()
out.release()
cv2.destroyAllWindows()
