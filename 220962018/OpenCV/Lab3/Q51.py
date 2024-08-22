import cv2
import numpy as np
from scipy.signal import correlate2d
import matplotlib.pyplot as plt

frame = cv2.imread('/home/student/220962018/OpenCV/Lab3/assets/Lenna_(test_image).png',0)
img_blur = frame

Kx = np.array(
    [[-1, 0, 1],
     [-2, 0, 2],
     [-1, 0, 1]], np.float32
)

Ky = np.array(
    [[-1, -2, -1],
     [0, 0, 0],
     [1, 2, 1]], np.float32
)

Gradient_Y = correlate2d(img_blur, Ky)
Gradient_X = correlate2d(img_blur, Kx)

fig = plt.figure()
plt.subplot(121)
plt.imshow(Gradient_Y, cmap='gray')
plt.title('Gradient$_y$')
plt.subplot(122)
plt.imshow(Gradient_X, cmap='gray')
plt.title('Gradient$_x$')

fig.set_figwidth(8)
plt.show()
plt.show()

def scale(x):
    return (x - x.min()) / (x.max() - x.min()) * 255

G = scale(np.hypot(Gradient_X, Gradient_Y))



EPS = np.finfo(float).eps # used to tackle the division by zero error
theta = np.arctan(Gradient_Y / (Gradient_X + EPS))

theta = np.arctan2(Gradient_Y, Gradient_X)

cv2.imshow('Combined Gradient',G)
cv2.waitKey(0)



def nms(G, theta):

    M, N = G.shape
    Z = np.zeros((M, N), dtype=np.int32)  # resultant image
    angle = theta * 180.0 / np.pi  # max -> 180, min -> -180
    angle[angle < 0] += 180  # max -> 180, min -> 0

    for i in range(1, M - 1):
        for j in range(1, N - 1):
            q = 255
            r = 255

            if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                r = G[i, j - 1]
                q = G[i, j + 1]

            elif 22.5 <= angle[i, j] < 67.5:
                r = G[i - 1, j + 1]
                q = G[i + 1, j - 1]

            elif 67.5 <= angle[i, j] < 112.5:
                r = G[i - 1, j]
                q = G[i + 1, j]

            elif 112.5 <= angle[i, j] < 157.5:
                r = G[i + 1, j + 1]
                q = G[i - 1, j - 1]

            if (G[i, j] >= q) and (G[i, j] >= r):
                Z[i, j] = G[i, j]
            else:
                Z[i, j] = 0
    return Z




def threshold(img, lowThresholdRatio=0.05, highThresholdRatio=0.09):
    '''
    Double threshold
    '''

    highThreshold = img.max() * highThresholdRatio;
    lowThreshold = highThreshold * lowThresholdRatio;

    M, N = img.shape
    res = np.zeros((M,N), dtype=np.int32)

    weak = 25
    strong = 255

    strong_i, strong_j = np.where(img >= highThreshold)
    # zeros_i, zeros_j = np.where(img < lowThreshold)

    weak_i, weak_j = np.where((img <= highThreshold) & (img >= lowThreshold))

    res[strong_i, strong_j] = strong
    res[weak_i, weak_j] = weak

    return res

res = threshold(nms(G,theta))




def hysteresis(img, weak, strong=255):
    M, N = img.shape

    for i in range(1, M-1):
        for j in range(1, N-1):
            if (img[i, j] == weak):
                if (
                    (img[i+1, j-1] == strong) or (img[i+1, j] == strong) or
                    (img[i+1, j+1] == strong) or (img[i, j-1] == strong) or
                    (img[i, j+1] == strong) or (img[i-1, j-1] == strong) or
                    (img[i-1, j] == strong) or (img[i-1, j+1] == strong)
                ):
                    img[i, j] = strong
                else:
                    img[i, j] = 0
    return img


