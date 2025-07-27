import cv2
import numpy as np

img = cv2.imread('image.png', cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (300, 300))
h, w = img.shape

kx = np.array([[-1, 0, 1],
               [-2, 0, 2],
               [-1, 0, 1]], dtype=np.float32)

ky = np.array([[-1, -2, -1],
               [ 0,  0,  0],
               [ 1,  2,  1]], dtype=np.float32)

gx = np.zeros_like(img, dtype=np.float32)
gy = np.zeros_like(img, dtype=np.float32)

for y in range(1, h-1):
    for x in range(1, w-1):
        patch = img[y-1:y+2, x-1:x+2]
        gx[y, x] = np.sum(patch * kx)
        gy[y, x] = np.sum(patch * ky)

mag = np.sqrt(gx*gx + gy*gy)
mag = np.clip(mag, 0, 255).astype(np.uint8)

cv2.imshow('Original', img)
cv2.imshow('Edges', mag)
cv2.waitKey(0)
cv2.destroyAllWindows()