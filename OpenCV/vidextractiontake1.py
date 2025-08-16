import cv2
import numpy as np
import time
from ultralytics import YOLO
import easyocr
import re
from clip_saver import GoalClipSaver  
from collections import Counter

model = YOLO("/media/dipen/New Volume/RnC Taskphase/OpenCV/best.pt")
video_path = "/media/dipen/New Volume/RnC Taskphase/OpenCV/testfootage.mp4"
reader = easyocr.Reader(['en'], gpu=True) 

cap = cv2.VideoCapture(video_path)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

PROCESS_EVERY_N_FRAMES = 3
OCR_EVERY_N_FRAMES = 3
frame_count = 0
cooldown_seconds = 8
last_goal_time = 0

scoreboard = []
time_history = []
extra_time_history = []
info_history = []
last_scoreboard_text = ""
last_time_text = ""
last_extra_time_text = ""
last_additional_info_text = ""
teams = ["BOU", "LIV", "MCI", "MUN", "TOT", "CHE", "ARS", "EVE", "WOL", "CRY",
         "NEW", "AVL", "BRE", "BHA", "BRN", "FUL", "LEE", "NFO", "SUN", "WHU"]

h, w = 300, 500
img = np.zeros((h, w, 3), dtype=np.uint8)

# ✅ Initialize clip saver
fps = cap.get(cv2.CAP_PROP_FPS)
clip_saver = GoalClipSaver(fps, pre_seconds=5, post_seconds=15)

goal_clip_count = 0
def detect_initial_score(video_path, model, reader, teams, frames_to_check=50, skip_frames=5):
    cap = cv2.VideoCapture(video_path)
    scores_seen = []
    hometeam, awayteam = "", ""

    frame_count = 0
    while len(scores_seen) < frames_to_check:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % skip_frames == 0:
            results = model(frame, verbose=False)
            scoreboard_roi, _, _, _ = extract_scoreboard_roi(results)
            if scoreboard_roi:
                detected_score = OcrResult(frame, scoreboard_roi)
                
                if detected_score:
                    half_index = len(detected_score) // 2
                    left_side = detected_score[:half_index].strip()
                    right_side = detected_score[half_index:].strip()

                    # detect teams
                    for team in teams:
                        if team in left_side and hometeam == "":
                            hometeam = team
                        if team in right_side and awayteam == "":
                            awayteam = team

                    # normalize score text
                    norm_text = normalize_score_text(detected_score, hometeam, awayteam)
                    scores_seen.append(norm_text)

        frame_count += 1

    cap.release()
    most_common_score = Counter(scores_seen).most_common(1)[0][0]
    print(f"📌 Initial detected scoreboard: {most_common_score}")

    # extract scores from text
    parts = most_common_score.split()
    home_score = int(parts[1])
    away_score = int(parts[2])
    return hometeam, awayteam, home_score, away_score

    if not scores_seen:
        return "", "", 0, 0
    
def ScoreDetect(scoreboard_detected, hometeam, awayteam, home_score, away_score):
    global last_goal_time, goal_clip_count
    detected_text = scoreboard_detected
    if "GOAL" in detected_text:
        now = time.time()
        if now - last_goal_time > cooldown_seconds:
            print("⚽ Goal detected!")
            last_goal_time = now

            # ✅ Save clip in background
            clip_saver.save_clip()
            goal_clip_count += 1

            if hometeam != "" and hometeam in detected_text:
                home_score += 1
            elif awayteam != "" and awayteam in detected_text:
                away_score += 1
    return home_score, away_score

# ==== Centered text drawing ====
def draw_centered_text(img, text, box_coords, font, scale, color, thickness):
    text = "" if text is None else str(text)
    (x1, y1), (x2, y2) = box_coords
    text_size = cv2.getTextSize(text, font, scale, thickness)[0]
    text_width, text_height = text_size
    x = x1 + (x2 - x1 - text_width) // 2
    y = y1 + (y2 - y1 + text_height) // 2
    cv2.putText(img, text, (x, y), font, scale, color, thickness, lineType=cv2.LINE_AA)


def clean_ocr_text(text):
    # Common OCR confusion corrections
    replacements = {
        'O': '0',
        'o': '0',
        'l': '1',
        'I': '1',
        '|': '1',
        '(': '0',
        'c': '0',
        'C': '0',
        'Uf': '0',  # Sometimes OCR merges chars
    }
    for wrong, right in replacements.items():
        text = text.replace(wrong, right)
    return text

def match_scoreboard_pattern(text):
    text = clean_ocr_text(text)
    # Matches TEAM score score TEAM (score = single digit)
    pattern = r"([A-Z]{3})\s\d\s\d\s([A-Z]{3})"
    match = re.search(pattern, text)
    if match:
        return match.groups()  # (team1, team2)
    return None

# ==== Draw live scoreboard ====
def display_scoreboard(team1, team2, score, match_time, extra_time, info):
    white, red, black = (255, 255, 255), (0, 0, 255), (0, 0, 0)
    cv2.rectangle(img, (0, 0), (w, h), white, -1)
    cv2.line(img, (0, 90), (w, 90), black, 2)
    cv2.line(img, (w//2, 0), (w//2, 90), black, 2)
    cv2.line(img, (0, 200), (w, 200), black, 2)
    cv2.line(img, (0, 250), (w, 250), black, 2)
    draw_centered_text(img, team1, ((0, 0), (w//2, 90)), cv2.FONT_HERSHEY_SIMPLEX, 1, black, 2)
    draw_centered_text(img, team2, ((w//2, 0), (w, 90)), cv2.FONT_HERSHEY_SIMPLEX, 1, black, 2)
    draw_centered_text(img, str(score), ((0, 90), (w, 200)), cv2.FONT_HERSHEY_SIMPLEX, 3, red, 3)
    draw_centered_text(img, f"{str(match_time)} {str(extra_time)}".strip(), ((0, 200), (w, 250)), cv2.FONT_HERSHEY_SIMPLEX, 1, red, 2)
    draw_centered_text(img, f"Info: {str(info)}", ((0, 250), (w, h)), cv2.FONT_HERSHEY_SIMPLEX, 1, black, 2)
    return img

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
        roi_gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2RGB)
        results = reader.readtext(roi_gray, detail=0)
        if results:
            return " ".join(results).strip()
    return ""

import re

def normalize_score_text(ocr_text, team1, team2):
    """
    Takes raw OCR scoreboard text and returns 'TEAM1 X X TEAM2'
    Example: 'WOL Uf CHE' -> 'WOL 0 1 CHE'
    """
    # Common OCR misreads mapped to digits
    digit_map = {
        'O': '0', 'o': '0', '(': '0', 'c': '0', '{': '0', '_': '0',
        'I': '1', 'l': '1', 'i': '1', 'f': '1', 'L': '1', 't': '1', 'u': '1', 'U': '1'
    }

    # Replace misreads
    fixed_text = ''.join(digit_map.get(ch, ch) for ch in ocr_text)

    # Extract digits only
    digits = re.findall(r'\d', fixed_text)

    # Pad if missing
    while len(digits) < 2:
        digits.append('0')

    score1, score2 = digits[0], digits[1]
    return f"{team1} {score1} {score2} {team2}"

hometeam, awayteam, home_score, away_score = detect_initial_score(video_path, model, reader, teams)
start_time = time.time()
processed_frames = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # ✅ Always add frame to clip_saver buffer
    clip_saver.add_frame(frame)

    frame_count += 1
    processed_frames += 1

    if frame_count % PROCESS_EVERY_N_FRAMES == 0:
        results = model(frame, verbose=False)
        scoreboard_roi, time_roi, extra_time_roi, additional_info_roi = extract_scoreboard_roi(results)

        if frame_count % OCR_EVERY_N_FRAMES == 0:
            if scoreboard_roi:
                detected_score = OcrResult(frame, scoreboard_roi)
                last_scoreboard_text = "" if detected_score is None else str(detected_score)
                home_score, away_score = ScoreDetect(detected_score, hometeam, awayteam, home_score, away_score)
                if "GOAL" in detected_score:
                    continue
                else:
                    print(f"Detected Score: {detected_score}")
                    last_scoreboard_text = normalize_score_text(detected_score, hometeam, awayteam)

            if time_roi:
                detected_time = OcrResult(frame, time_roi)
                last_time_text = "" if detected_time is None else str(detected_time)

            if extra_time_roi:
                detected_extra = OcrResult(frame, extra_time_roi)
                last_extra_time_text = "" if detected_extra is None else str(detected_extra)

            if additional_info_roi:
                detected_info = OcrResult(frame, additional_info_roi)
                last_additional_info_text = "" if detected_info is None else str(detected_info)

            print(f"{frame_count} Scoreboard: {last_scoreboard_text}")

        annotated_frame = results[0].plot()
    else:
        annotated_frame = frame.copy()

    annotated_frame = cv2.resize(annotated_frame, (640, 480))
    scoreboard_img = display_scoreboard(hometeam, awayteam, f"{home_score}-{away_score}", last_time_text, last_extra_time_text, last_additional_info_text)
    cv2.imshow("Match Details", scoreboard_img)
    cv2.imshow("Frame", annotated_frame)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

total_time = time.time() - start_time
print(f"Average FPS: {processed_frames / total_time:.2f}")
print(f"Final Score: {hometeam} {home_score} - {away_score} {awayteam}")
