import cv2
import numpy as np
import matplotlib.pyplot as plt

def detect_edges(image_path, low_threshold=50, high_threshold=150):
    # Load the image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Image at path {image_path} not found.")

    # Apply Gaussian blur to reduce noise and improve edge detection
    blurred_image = cv2.GaussianBlur(image, (5, 5), 0)

    # Apply Canny edge detector
    edges = cv2.Canny(blurred_image, low_threshold, high_threshold)

    # Display results
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.title('Original Image')
    plt.imshow(image, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title('Detected Edges')
    plt.imshow(edges, cmap='gray')
    plt.axis('off')

    plt.show()

# Example usage
detect_edges('/home/student/220962018/OpenCV/Lab3/assets/4983606056845312.png')
