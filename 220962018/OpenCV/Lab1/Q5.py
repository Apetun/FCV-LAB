import cv2
img = cv2.imread("./assets/images.jpeg")
ans = cv2.resize(img, (1080, 1080))
cv2.imwrite("./assets/images2.jpeg", ans)
cv2.waitKey(0)
