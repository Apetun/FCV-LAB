import cv2

img = cv2.imread("./assets/images.jpeg")
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

x = int(input("Enter the x value: "))
y = int(input("Enter the y value: "))
print("RGB values are :")
print(rgb[x, y, 0], rgb[x, y, 1], rgb[x, y, 2])
