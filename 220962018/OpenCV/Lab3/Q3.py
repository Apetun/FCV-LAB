import cv2
import matplotlib.pyplot as plt

def apply_filters(image_path):
    # Load the image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Image at path {image_path} not found.")

    # Apply Box filter (average filter)
    box_filter = cv2.blur(image, (5, 5))  # 5x5 box filter

    # Apply Gaussian filter
    gaussian_filter = cv2.GaussianBlur(image, (5, 5), sigmaX=0)

    # Display results
    plt.figure(figsize=(12, 8))

    plt.subplot(1, 3, 1)
    plt.title('Original Image')
    plt.imshow(image, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title('Box Filter')
    plt.imshow(box_filter, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title('Gaussian Filter')
    plt.imshow(gaussian_filter, cmap='gray')
    plt.axis('off')

    plt.show()

# Example usage
apply_filters('/home/student/220962018/OpenCV/Lab3/assets/4983606056845312.png')
