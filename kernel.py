import cv2
import numpy as np

img = cv2.imread("img/image.png", cv2.IMREAD_GRAYSCALE) 
width = 300
height = 300
img = cv2.resize(img,(width,height))
kernel = np.array([[0, -1,  0],
                   [-1, 5, -1],
                   [0, -1,  0]], dtype=np.float32)  
print(type(img))
h, w = img.shape
out = np.zeros_like(img, dtype=np.float32)

for y in range(1, h-1):
    for x in range(1, w-1):
        patch = img[y-1:y+2, x-1:x+2]
        out[y, x] = np.sum(patch * kernel)

out = np.clip(out, 0, 255).astype(np.uint8)
print(out)
cv2.imshow("original", img)
cv2.imshow("filtered", out)
cv2.waitKey(0)
cv2.destroyAllWindows()