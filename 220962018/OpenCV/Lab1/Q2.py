import cv2

cap = cv2.VideoCapture('./assets/All 3 default dances.mp4')
out = cv2.VideoWriter()

while cap.isOpened():
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    out.write(frame)
    cv2.imshow("Test", gray)
    if cv2.waitKey(1) == ord('q'):
        break
