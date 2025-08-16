import cv2

cap= cv2.VideoCapture("D:\RnC Taskphase\OpenCV\SourceVideos\\testingvid.mp4")

while True:
    _, frame = cap.read()

    gray= cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur= cv2.GaussianBlur(gray, (3,3), 0)
    edges= cv2.Canny(blur, 50, 150)
    dilated= cv2.dilate(edges, None, iterations=1)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)
    cv2.imshow('Contours', frame)
    cv2.imshow('Edges', edges)
    cv2.imshow('Blur', blur)
    cv2.imshow('Dilated', dilated)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()