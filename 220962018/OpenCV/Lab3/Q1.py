import cv2
import numpy as np


def unsharp_masking(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Unable to load image.")
        return
    blurred = cv2.GaussianBlur(image, (7, 7), 0)
    img = np.asarray(image)
    gaussian = np.asarray(blurred)
    mask = img - gaussian
    sharpened = image + mask

    cv2.imshow('Sharpened Image', sharpened)
    cv2.waitKey(0)



if __name__ == "__main__":
    input_image_path = './assets/4983606056845312.png'
    unsharp_masking(input_image_path)
