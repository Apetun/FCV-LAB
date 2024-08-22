import cv2
import numpy as np
import matplotlib.pyplot as plt


def compute_gradient(image_path):
    # Load the image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Image at path {image_path} not found.")

    # Compute gradients in x and y direction using Sobel operator
    grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

    # Compute gradient magnitude
    gradient_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)

    # Compute gradient direction
    gradient_direction = np.arctan2(grad_y, grad_x) * (180 / np.pi) % 360

    # Display results
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 3, 1)
    plt.title('Original Image')
    plt.imshow(image, cmap='gray')

    plt.subplot(1, 3, 2)
    plt.title('Gradient Magnitude')
    plt.imshow(gradient_magnitude, cmap='gray')

    plt.subplot(1, 3, 3)
    plt.title('Gradient Direction')
    plt.imshow(gradient_direction, cmap='hsv')

    plt.show()


# Example usage
compute_gradient('/home/student/220962018/OpenCV/Lab3/assets/4983606056845312.png')
