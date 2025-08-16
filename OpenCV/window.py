import cv2
import numpy as np

def draw_centered_text(img, text, box_coords, font, scale, color, thickness):
    (x1, y1), (x2, y2) = box_coords
    text_size = cv2.getTextSize(text, font, scale, thickness)[0]
    text_width, text_height = text_size
    x = x1 + (x2 - x1 - text_width) // 2
    y = y1 + (y2 - y1 + text_height) // 2
    cv2.putText(img, text, (x, y), font, scale, color, thickness)

# Create a blank image
img = np.zeros((300, 500, 3), dtype=np.uint8)

# Data
Team1 = "Team A"
Team2 = "Team B"
Score_A=2
Score_B=1
time = "78:45"
extra_time = "+3"
additional_info = "No red cards"
teams=["BOU", "LIV", "MCI", "MUN", "TOT", "CHE", "ARS", "EVE", "WOL", "CRY",
       "NEW", "AVL", "BRE", "BHA", "BRN", "FUL", "LEE", "NFO", "SUN", "WHU"]

# Colors
white = (255, 255, 255)
red = (0, 0, 255)
black = (0, 0, 0)

# Background
cv2.rectangle(img, (0, 0), (500, 300), white, -1)

# Layout lines
cv2.line(img, (0, 90), (500, 90), black, 2)   # after team row
cv2.line(img, (250, 0), (250, 90), black, 2)  # between team boxes
cv2.line(img, (0, 200), (500, 200), black, 2) # after time row
cv2.line(img, (0, 250), (500, 250), black, 2) # after time row

# Draw centered text
draw_centered_text(img, Team1, ((0, 0), (250, 90)), cv2.FONT_HERSHEY_SIMPLEX, 1, black, 2)
draw_centered_text(img, Team2, ((250, 0), (500, 90)), cv2.FONT_HERSHEY_SIMPLEX, 1, black, 2)
draw_centered_text(img, f"{Score_A} - {Score_B}", ((0, 90), (500, 200)), cv2.FONT_HERSHEY_SIMPLEX, 3, red, 3)
draw_centered_text(img, f"{time} {extra_time}", ((0, 200), (500, 250)), cv2.FONT_HERSHEY_SIMPLEX, 1, red, 2)
draw_centered_text(img, f"Info: {additional_info}", ((0, 250), (500, 300)), cv2.FONT_HERSHEY_SIMPLEX, 1, black, 2)

# Show
cv2.imshow("Match Details", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
