import cv2
import numpy as np
import time
import os
from ultralytics import YOLO
import easyocr
from collections import deque

# =========================
# Simple Clip Saver Class
# =========================
class GoalClipSaver:
    def __init__(self, fps, pre_seconds=5, post_seconds=15, save_dir="goal_clips", video_duration=None):
        self.fps = fps
        self.pre_frames = int(pre_seconds * fps)
        self.post_frames = int(post_seconds * fps)
        self.buffer = deque(maxlen=self.pre_frames)
        self.saving = False
        self.frames_left_to_save = 0
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.writer = None
        self.clip_index = 0
        self.video_duration = video_duration  # total frames in video (optional)

    def add_frame(self, frame):
        if not self.saving:
            self.buffer.append(frame.copy())
        else:
            if self.frames_left_to_save > 0:
                self.writer.write(frame)
                self.frames_left_to_save -= 1
            else:
                # Finish clip early if no more frames
                self.writer.release()
                self.writer = None
                self.saving = False
                print(f"✅ Clip saved: {self.last_clip_path}")

    def save_clip(self):
        if self.saving:
            return  # already saving

        self.saving = True
        self.frames_left_to_save = self.post_frames

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.last_clip_path = os.path.join(self.save_dir, f"highlight_{timestamp}.mp4")

        if not self.buffer:
            print("⚠ No pre-goal frames available, starting from current frame.")
            h, w = (720, 1280)  # fallback size if buffer empty
        else:
            h, w = self.buffer[0].shape[:2]

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer = cv2.VideoWriter(self.last_clip_path, fourcc, self.fps, (w, h))

        # Pre-goal frames (if any)
        for f in self.buffer:
            self.writer.write(f)

        print(f"💾 Saving clip to {self.last_clip_path}")

# =========================
# Your Existing Script
# =========================
model = YOLO("/media/dipen/New Volume/RnC Taskphase/OpenCV/best(1).pt")
video_path = "/media/dipen/New Volume/RnC Taskphase/OpenCV/fullmatch2.mp4"
cap = cv2.VideoCapture(video_path)

reader = easyocr.Reader(['en'], gpu=True)
PROCESS_EVERY_N_FRAMES = 3
OCR_EVERY_N_FRAMES = 3
frame_count = 0
cooldown_seconds = 8
last_goal_time = 0
hometeam = ""
awayteam = ""
home_score = 0
away_score = 0
match_phase = ""
teams = ["BOU", "LIV", "MCI", "MUN", "TOT", "CHE", "ARS", "EVE", "WOL", "CRY",
         "NEW", "AVL", "BRE", "BHA", "BRN", "FUL", "LEE", "NFO", "SOU", "WHU"]

img = np.zeros((300, 500, 3), dtype=np.uint8)
h, w = img.shape[:2]

fps = cap.get(cv2.CAP_PROP_FPS)
clip_saver = GoalClipSaver(fps, pre_seconds=7, post_seconds=15)

def draw_centered_text(img, text, box_coords, font, scale, color, thickness):
    text = "" if text is None else str(text)
    (x1, y1), (x2, y2) = box_coords
    text_size = cv2.getTextSize(text, font, scale, thickness)[0]
    text_width, text_height = text_size
    x = x1 + (x2 - x1 - text_width) // 2
    y = y1 + (y2 - y1 + text_height) // 2
    cv2.putText(img, text, (x, y), font, scale, color, thickness, lineType=cv2.LINE_AA)

def display_scoreboard(team1, team2, score, phase):
    white, red, black = (255, 255, 255), (0, 0, 255), (0, 0, 0)
    cv2.rectangle(img, (0, 0), (w, h), white, -1)
    cv2.line(img, (0, 90), (w, 90), black, 2)
    cv2.line(img, (w//2, 0), (w//2, 90), black, 2)
    cv2.line(img, (0, 200), (w, 200), black, 2)
    draw_centered_text(img, team1, ((0, 0), (w//2, 90)), cv2.FONT_HERSHEY_SIMPLEX, 1, black, 2)
    draw_centered_text(img, team2, ((w//2, 0), (w, 90)), cv2.FONT_HERSHEY_SIMPLEX, 1, black, 2)
    draw_centered_text(img, str(score), ((0, 90), (w, 200)), cv2.FONT_HERSHEY_SIMPLEX, 3, red, 3)
    draw_centered_text(img, phase, ((0, 200), (w, h)), cv2.FONT_HERSHEY_SIMPLEX, 1, black, 2)
    return img

def extract_scoreboard_roi(results):
    scoreboard_rois, additional_info_rois = [], []
    for r in results:
        boxes = r.boxes.xyxy
        clses = r.boxes.cls
        for box, cls in zip(boxes, clses):
            x1, y1, x2, y2 = box.tolist()
            name = model.names[int(cls)]
            if name == "Scoreboard":
                scoreboard_rois.append((x1, y1, x2, y2))
            elif name == "additonalinfo":
                additional_info_rois.append((x1, y1, x2, y2))
    return scoreboard_rois, additional_info_rois

def OcrResult(img, roi_list):
    if roi_list:
        x1, y1, x2, y2 = map(int, roi_list[0])
        roi_frame = img[y1:y2, x1:x2]
        roi_rgb = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2RGB)
        results = reader.readtext(roi_rgb, detail=0)
        if results:
            return " ".join(results).strip()
    return ""

def ScoreDetect(scoreboard_detected, hometeam, awayteam, home_score, away_score):
    global last_goal_time
    if "goal" in scoreboard_detected.lower():
        now = time.time()
        print(f"Detected Scoreboard: {scoreboard_detected}")
        if now - last_goal_time > cooldown_seconds:
            print("Goal detected! Saving highlight...")
            last_goal_time = now
            clip_saver.save_clip()  # Save clip (5 sec before, 15 sec after)
            if hometeam!="" and hometeam.lower() in scoreboard_detected.lower():
                home_score += 1
            elif awayteam!="" and awayteam.lower() in scoreboard_detected.lower():
                away_score += 1
    if "yellow card" in scoreboard_detected.lower():
        print("Yellow card detected!")
        clip_saver.save_clip()
    if "red card" in scoreboard_detected.lower():
        print("Red card detected!")
        clip_saver.save_clip()
    return home_score, away_score

while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        continue

    clip_saver.add_frame(frame) 
    frame_count += 1

    if frame_count % PROCESS_EVERY_N_FRAMES == 0:
        results = model(frame, verbose=False)
        scoreboard_roi, additional_info_roi = extract_scoreboard_roi(results)

        if frame_count % OCR_EVERY_N_FRAMES == 0:
            if scoreboard_roi:
                detected_score = OcrResult(frame, scoreboard_roi)
                half_index = len(detected_score) // 2
                left_side = detected_score[:half_index].strip()
                right_side = detected_score[half_index:].strip()

                for team in teams:
                    if team in left_side and not hometeam:
                        hometeam = team
                    if team in right_side and not awayteam:
                        awayteam = team

                home_score, away_score = ScoreDetect(detected_score, hometeam, awayteam, home_score, away_score)

            if additional_info_roi:
                detected_info = OcrResult(frame, additional_info_roi).lower()
                if "first half" in detected_info:
                    match_phase = "First Half"
                elif "half-time" in detected_info:
                    match_phase = "Half-Time"
                elif "second half" in detected_info:
                    match_phase = "Second Half"
                elif "full time" in detected_info:
                    match_phase = "Full Time"

            print(f"{frame_count} | Scoreboard: {hometeam} {home_score}-{away_score} {awayteam} | Phase: {match_phase}")

        annotated_frame = results[0].plot()
    else:
        annotated_frame = frame.copy()

    scoreboard_img = display_scoreboard(hometeam, awayteam, f"{home_score}-{away_score}", match_phase)
    cv2.imshow("Match Details", scoreboard_img)
    cv2.imshow("Frame", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print(f"Final Score: {hometeam} {home_score} - {away_score} {awayteam}")
