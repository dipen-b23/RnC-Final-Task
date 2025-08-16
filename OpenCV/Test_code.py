from ultralytics import YOLO
import cv2
import easyocr
import numpy as np

model = YOLO("/media/dipen/New Volume/RnC Taskphase/OpenCV/best (1).pt")
classes = model.names
reader = easyocr.Reader(['en'], gpu=True)
window=np.zeros((100, 250, 3), np.uint8)

# === 2. ROI extractor ===
def extract_scoreboard_roi(results):
    scoreboard_rois, time_rois, extra_time_rois, additional_info_rois = [], [], [], []
    for r in results:
        boxes = r.boxes.xyxy
        confs = r.boxes.conf
        clses = r.boxes.cls
        for box, conf, cls in zip(boxes, confs, clses):
            x1, y1, x2, y2 = box.tolist()
            name = model.names[int(cls)]
            if name == "Scoreboard":
                scoreboard_rois.append(((x1+7), (y1+7), (x2+7), (y2+7)))
            elif name == "time":
                time_rois.append(((x1+7), (y1+7), (x2+7), (y2+7)))
            elif name == "extra_time":
                extra_time_rois.append(((x1+7), (y1+7), (x2+7), (y2+7)))
            elif name == "additonalinfo":
                additional_info_rois.append(((x1+7), (y1+7), (x2+7), (y2+7)))
    return scoreboard_rois, time_rois, extra_time_rois, additional_info_rois


def OcrResult(img, roi_list):
    if roi_list:
        x1, y1, x2, y2 = map(int, roi_list[0])  # Take first ROI
        roi_frame = img[y1:y2, x1:x2]
        roi_rgb = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
        canny = cv2.Canny(roi_rgb, 50, 150)
        _, thresh = cv2.threshold(canny, 150, 255, cv2.THRESH_BINARY)
        results = reader.readtext(roi_rgb, detail=0)  # detail=0 returns just text strings
        if results:
            return " ".join(results).strip()
    return ""
def display_scoreboard(score, match_time, extra_time, info):
    h, w = 300, 600
    scoreboard_img = np.zeros((h, w, 3), dtype=np.uint8)  # black background

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    color = (255, 255, 255)  
    thickness = 2
    cv2.putText(scoreboard_img, f"Score: {score}", (20, 50), font, font_scale, color, thickness)
    cv2.putText(scoreboard_img, f"Time: {match_time}", (20, 100), font, font_scale, color, thickness)
    cv2.putText(scoreboard_img, f"Extra Time: {extra_time}", (20, 150), font, font_scale, color, thickness)
    cv2.putText(scoreboard_img, f"Info: {info}", (20, 200), font, font_scale, color, thickness)
    cv2.imshow("Scoreboard", scoreboard_img)


# === 4. Process video ===
video_path = "/media/dipen/New Volume/RnC Taskphase/OpenCV/SourceVideos/prediction of big scoreboard.mp4"
cap = cv2.VideoCapture(video_path)
i=0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO detection
    results = model(frame, verbose=False)

    # Extract ROIs
    scoreboard_roi, time_roi, extra_time_roi, additional_info_roi = extract_scoreboard_roi(results)
    # OCR
    scoreboard_text = OcrResult(frame, scoreboard_roi)
    time_text = OcrResult(frame, time_roi)
    extra_time_text = OcrResult(frame, extra_time_roi)
    additional_info_text = OcrResult(frame, additional_info_roi)
    i+=1
    # Show live results
    print(f"{(i)} Scoreboard: {scoreboard_text} | Time: {time_text} | Extra: {extra_time_text} | Info: {additional_info_text.strip()}")

    annotated_frame = results[0].plot()
    # annotated_frame.resize(640, 480)
    display_scoreboard(scoreboard_text, time_text, extra_time_text, additional_info_text)
    cv2.imshow("YOLO Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
