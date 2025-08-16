import cv2
from ultralytics import YOLO
# import pytesseract
import easyocr
import re

model = YOLO("/media/dipen/New Volume/RnC Taskphase/OpenCV/best.pt")

# Optional: Tesseract path for Windows users
# pytesseract.pytesseract.tesseract_cmd = r"/usr/bin/tesseract"

frame = cv2.imread("/media/dipen/New Volume/RnC Taskphase/OpenCV/FramesForDataset/frame_9270.jpg")
cv2.rectangle(frame, (80,53), (160,75), (0, 0, 0), -1)
results = model(frame)
hometeam = ""
awayteam = ""
home_score = 0
away_score = 0
extracted_text = ""
teams = ["BOU", "LIV", "MCI", "MUN", "TOT", "CHE", "ARS", "EVE", "WOL", "CRY",
         "NEW", "AVL", "BRE", "BHA", "BRN", "FUL", "LEE", "NFO", "SUN", "WHU"]

for result in results:
    for box in result.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        
        if cls == 0 and conf > 0.5:  # Scoreboard text
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            scoreboard_crop = frame[y1:y2, x1:x2]

            # Using pytesseract for team names/scores is tricky, keeping EasyOCR here is fine
            import easyocr
            reader = easyocr.Reader(['en'], gpu=True)
            ocr_results = reader.readtext(scoreboard_crop)
            
            detected_text = " ".join([text for _, text, _ in ocr_results]).upper()
            extracted_text = detected_text
            print(f"Detected Text: {detected_text}")
            if "GOAL" in detected_text:
                print("⚽ Goal detected!")
            half_index = len(detected_text) // 2
            left_side = detected_text[:half_index].strip()
            right_side = detected_text[half_index:].strip()

            for team in teams:
                if team in left_side and hometeam == "":
                    hometeam = team
                if team in right_side and awayteam == "":
                    awayteam = team
            if "GOAL" in detected_text:
                if hometeam != "" and hometeam in detected_text:
                    home_score += 1
                if awayteam != "" and awayteam in detected_text:
                    away_score += 1

        if cls == 3 and conf > 0.5:  # Time box
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            time_crop = frame[y1:y2, x1:x2]

            # Preprocess for better OCR
            time_gray = cv2.cvtColor(time_crop, cv2.COLOR_BGR2GRAY)
            time_gray = cv2.threshold(time_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

            # EasyOCR detection (only numbers + colon)
            result = reader.readtext(time_gray, allowlist='0123456789:', detail=0)
            detected_time = result[0] if result else ''
            
            # Validate with regex
            match = re.search(r'\d{1,2}:\d{2}', detected_time)
            if match:
                print(f"Time Detected: {match.group()}")
            else:
                print(f"Time OCR raw: {detected_time} (no valid match)")

        if cls == 2 and conf > 0.5:  # Extra time box
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            extra_time_crop = frame[y1:y2, x1:x2]

            extra_time_gray = cv2.cvtColor(extra_time_crop, cv2.COLOR_BGR2GRAY)
            extra_time_gray = cv2.threshold(extra_time_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

            result = reader.readtext(extra_time_gray, allowlist='0123456789:', detail=0)
            extra_time_text = result[0] if result else ''
            
            print(f"Extra Time Detected: {extra_time_text}")
annotated_frame = results[0].plot()
cv2.imshow("Frame", annotated_frame)
cv2.waitKey(0)
cv2.destroyAllWindows()

print(f"Home Team: {hometeam}, Away Team: {awayteam}")
print(f"Home Score: {home_score}, Away Score: {away_score}")
print(f"Extracted Text: {extracted_text}")
