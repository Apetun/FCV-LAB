import cv2
import numpy as np
import matplotlib.pyplot as plt

img_array = cv2.imread("./assets/source.jpg", 0)

hist = np.bincount(img_array.flatten(), minlength=256)
pixsum = np.sum(hist)
hist = hist / pixsum

chist = np.cumsum(hist)
transform_map = np.floor(255 * chist).astype(np.uint8)
eq_img = transform_map[img_array]

eq_hist = np.bincount(eq_img.flatten(), minlength=256)
eq_pixsum = np.sum(eq_hist)
eq_hist = eq_hist / eq_pixsum

cum_hist = np.cumsum(hist)
cum_eq_hist = np.cumsum(eq_hist)

max_histogram_height = max(np.max(hist), np.max(eq_hist))
norm_histogram_array = hist / max_histogram_height
norm_eq_histogram_array = eq_hist / max_histogram_height

plt.figure(figsize=(14, 10))

plt.subplot(2, 2, 1)
plt.title('Original Image')
plt.imshow(img_array, cmap='gray')
plt.axis('off')

plt.subplot(2, 2, 2)
plt.title('Original Histogram with CDF')
plt.bar(range(256), norm_histogram_array, color='red', alpha=0.7, label='Histogram')
plt.plot(range(256), cum_hist, color='blue', label='CDF', linestyle='--')
plt.xlim(-5, 260)
plt.ylim(-0.05, 1.05)
plt.legend()

plt.subplot(2, 2, 3)
plt.title('Equalized Image')
plt.imshow(eq_img, cmap='gray')
plt.axis('off')

plt.subplot(2, 2, 4)
plt.title('Equalized Histogram with CDF')
plt.bar(range(256), norm_eq_histogram_array, color='red', alpha=0.7, label='Histogram')
plt.plot(range(256), cum_eq_hist, color='blue', label='CDF', linestyle='--')
plt.xlim(-5, 260)
plt.ylim(-0.05, 1.05)
plt.legend()

plt.tight_layout()
plt.subplots_adjust(wspace=0.3, hspace=0.3)

plt.show()
