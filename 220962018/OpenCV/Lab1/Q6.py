import cv2

img = cv2.imread("./assets/images.jpeg")
cv2.imshow("Rotated Img", cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE))
cv2.waitKey(0)
