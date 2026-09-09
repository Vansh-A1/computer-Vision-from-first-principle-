import cv2
import numpy as np


image_path = "/data/projectwork/cv/Pasted image.png"


img = cv2.imread(image_path)

if img is None:
    raise FileNotFoundError(f"Could not load image: {image_path}")


gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = np.float32(gray)

corners = cv2.cornerHarris(
    gray,
    blockSize=2,
    ksize=3,
    k=0.04
)

corners = cv2.dilate(corners, None)

threshold = 0.01 * corners.max()
img[corners > threshold] = [0, 0, 255]

output_path = "/data/projectwork/cv/harris_corners.png"
cv2.imwrite(output_path, img)

print(f"Saved corner detection result to: {output_path}")
