from ultralytics import YOLO
import cv2
import json
from collections import defaultdict
from statistics import mean

# Modeli yükleyin
model = YOLO("dataaugm.pt")

# Video dosyasını açın
video_path = r"C:\Users\cagla\OneDrive\Masaüstü\yoloproje\video.mp4"
cap = cv2.VideoCapture(video_path)

# Video özelliklerini alın
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Video kaydedici tanımlayın
output_path = r"C:\Users\cagla\OneDrive\Masaüstü\yoloproje\output_video.mp4"
out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

# JSON dosyasının yolu
json_output_path_avg_score_frequent_label = r"C:\Users\cagla\OneDrive\Masaüstü\yoloproje\output_data_avg_score_frequent_label.json"

# Sınıf isimleri (bu modelinize göre değişebilir, örneğin COCO dataseti için)
class_names = model.names

# ID'ler için etiket ve skor bilgilerini saklamak için dictionary
id_label_scores = defaultdict(lambda: defaultdict(list))
# Nesnelerin sabit skorlarını saklamak için dictionary
fixed_scores = {}

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Kare üzerinde tahmin yapın
    results = model.track(frame, persist=True, show=False, tracker="botsort.yaml")

    # Tahmin edilen kutuları al
    boxes = results[0].boxes.xyxy  # Çerçevelerin xyxy formatında koordinatları
    confidences = results[0].boxes.conf  # Güven değerleri
    class_ids = results[0].boxes.cls  # Sınıf id'leri
    track_ids = results[0].boxes.id  # İzleme id'leri

    # Her bir tahmin edilen çerçeveyi çizdirin
    for box, conf, class_id, track_id in zip(boxes, confidences, class_ids, track_ids):
        x1, y1, x2, y2 = map(int, box)
        class_name = class_names[int(class_id)] if class_id < len(class_names) else "Other"  # Sınıf adı
        track_id = int(track_id) if track_id is not None else -1

        if track_id not in fixed_scores:
            # Eğer bu ID için sabit skor yoksa, ilk tespit edilen skoru kullan
            fixed_scores[track_id] = (class_name, conf)

        # Sabit skoru al
        class_name, fixed_conf = fixed_scores[track_id]
        label = f"{class_name}: {fixed_conf:.2f}"
        color = (0, 255, 0)  # Kutular için yeşil renk kullanıyoruz

        # Çerçeveyi çiz
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Metni çiz
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # Annotasyonlu kareyi yaz
    out.write(frame)

    # İşlenen kareyi göster
    cv2.imshow("YOLOv8 Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Kaynakları serbest bırakın
cap.release()
out.release()
cv2.destroyAllWindows()

# Ortalama skoru ve en sık görülen label'ı hesaplama
results_data = []
for track_id, (class_name, fixed_conf) in fixed_scores.items():
    results_data.append({
        "ProductRecognition": {
            "name": class_name,
            "average_score": fixed_conf,
            "id": track_id
        }
    })

# JSON dosyasını kaydetme
with open(json_output_path_avg_score_frequent_label, "w") as json_file:
    json.dump(results_data, json_file, indent=4)
