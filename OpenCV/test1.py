import cv2
import numpy as np
import time
from ultralytics import YOLO
import easyocr
teams=["BOU", "LIV", "MCI", "MUN", "TOT", "CHE", "ARS", "EVE", "WOL", "CRY",
       "NEW", "AVL", "BRE", "BHA", "BRN", "FUL", "LEE", "NFO", "SUN", "WHU"]

# ==== Centered text drawing ====
def draw_centered_text(img, text, box_coords, font, scale, color, thickness):
    (x1, y1), (x2, y2) = box_coords
    text_size = cv2.getTextSize(text, font, scale, thickness)[0]
    text_width, text_height = text_size
    x = x1 + (x2 - x1 - text_width) // 2
    y = y1 + (y2 - y1 + text_height) // 2
    cv2.putText(img, text, (x, y), font, scale, color, thickness)

# ==== Draw live scoreboard ====
def display_scoreboard(team1, team2, score, match_time, extra_time, info):
    h, w = 300, 500
    img = np.zeros((h, w, 3), dtype=np.uint8)
    white = (255, 255, 255)
    red = (0, 0, 255)
    black = (0, 0, 0)

    # Background
    cv2.rectangle(img, (0, 0), (w, h), white, -1)

    # Layout lines
    cv2.line(img, (0, 90), (w, 90), black, 2)      # after team row
    cv2.line(img, (w//2, 0), (w//2, 90), black, 2) # between team boxes
    cv2.line(img, (0, 200), (w, 200), black, 2)    # after score row
    cv2.line(img, (0, 250), (w, 250), black, 2)    # after time row

    # Draw centered text
    draw_centered_text(img, team1, ((0, 0), (w//2, 90)), cv2.FONT_HERSHEY_SIMPLEX, 1, black, 2)
    draw_centered_text(img, team2, ((w//2, 0), (w, 90)), cv2.FONT_HERSHEY_SIMPLEX, 1, black, 2)
    draw_centered_text(img, score, ((0, 90), (w, 200)), cv2.FONT_HERSHEY_SIMPLEX, 3, red, 3)
    draw_centered_text(img, f"{match_time} {extra_time}", ((0, 200), (w, 250)), cv2.FONT_HERSHEY_SIMPLEX, 1, red, 2)
    draw_centered_text(img, f"Info: {info}", ((0, 250), (w, h)), cv2.FONT_HERSHEY_SIMPLEX, 1, black, 2)

    cv2.imshow("Match Details", img)

# ==== YOLO + OCR setup ====
model = YOLO("/media/dipen/New Volume/RnC Taskphase/OpenCV/best (1).pt")
reader = easyocr.Reader(['en'], gpu=True)

PROCESS_EVERY_N_FRAMES = 3
OCR_EVERY_N_FRAMES = 6
frame_count = 0

scoreboard=[]
last_scoreboard_text = ""
last_time_text = ""
last_extra_time_text = ""
last_additional_info_text = ""

score_team1 = 0
score_team2 = 0

# === ROI extractor ===
def extract_scoreboard_roi(results):
    scoreboard_rois, time_rois, extra_time_rois, additional_info_rois = [], [], [], []
    for r in results:
        boxes = r.boxes.xyxy
        clses = r.boxes.cls
        for box, cls in zip(boxes, clses):
            x1, y1, x2, y2 = box.tolist()
            name = model.names[int(cls)]
            if name == "Scoreboard":
                scoreboard_rois.append((x1+7, y1+7, x2+7, y2+7))
            elif name == "time":
                time_rois.append((x1+7, y1+7, x2+7, y2+7))
            elif name == "extra_time":
                extra_time_rois.append((x1+7, y1+7, x2+7, y2+7))
            elif name == "additonalinfo":
                additional_info_rois.append((x1+7, y1+7, x2+7, y2+7))
    return scoreboard_rois, time_rois, extra_time_rois, additional_info_rois

# === OCR helper ===
def OcrResult(img, roi_list):
    if roi_list:
        x1, y1, x2, y2 = map(int, roi_list[0])
        roi_frame = img[y1:y2, x1:x2]
        roi_gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
        canny = cv2.Canny(roi_gray, 50, 150)
        _, thresh = cv2.threshold(canny, 150, 255, cv2.THRESH_BINARY)
        results = reader.readtext(thresh, detail=0)
        if results:
            return " ".join(results).strip()
    return ""

# ==== Video processing ====
video_path = "/media/dipen/New Volume/RnC Taskphase/OpenCV/SampleVideo.mp4"
cap = cv2.VideoCapture(video_path)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

start_time = time.time()
processed_frames = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    processed_frames += 1

    if frame_count % PROCESS_EVERY_N_FRAMES == 0:
        results = model(frame, verbose=False)
        scoreboard_roi, time_roi, extra_time_roi, additional_info_roi = extract_scoreboard_roi(results)

        if frame_count % OCR_EVERY_N_FRAMES == 0:
            if scoreboard_roi:
                last_scoreboard_text = OcrResult(frame, scoreboard_roi)
                scoreboard.append(OcrResult(frame, scoreboard_roi))
                for team in teams:
                    if team.lower() in last_scoreboard_text.lower():
                        if "goal" in last_scoreboard_text.lower():
                            if team == "Team A":
                                score_team1 += 1
                            elif team == "Team B":
                                score_team2 += 1
                # if "goal" in last_scoreboard_text.lower() :
                #     score_team1 += 1 
                    
            if time_roi:
                last_time_text = OcrResult(frame, time_roi)
            if extra_time_roi:
                last_extra_time_text = OcrResult(frame, extra_time_roi)
            if additional_info_roi:
                last_additional_info_text = OcrResult(frame, additional_info_roi)

        annotated_frame = results[0].plot()
        if frame_count % OCR_EVERY_N_FRAMES == 0:
            print(f"{frame_count} Scoreboard: {last_scoreboard_text} | Time: {last_time_text} | Extra: {last_extra_time_text} | Info: {last_additional_info_text}")
    else:
        annotated_frame = frame.copy()

    annotated_frame = cv2.resize(annotated_frame, (640, 480))

    # Show both annotated frame & custom scoreboard
    display_scoreboard("Team A", "Team B", last_scoreboard_text, last_time_text, last_extra_time_text, last_additional_info_text)
    cv2.imshow("Frame", annotated_frame)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

total_time = time.time() - start_time
print(f"Average FPS: {processed_frames / total_time:.2f}")
print(f"Total frames processed: {processed_frames}")
print(f"Total time: {total_time:.2f} seconds")
