import cv2
img = cv2.imread("./assets/images.jpeg", 0)
cv2.imshow("Image", img)
cv2.imwrite("./assets/images1.jpeg", img)
cv2.waitKey(0)
