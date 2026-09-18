import cv2
import numpy as np

# ----------------------------
# 1. Load Image
# ----------------------------
img = cv2.imread("image.jpg")

if img is None:
    raise FileNotFoundError("Could not find image.jpg")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = np.float32(gray)


# ----------------------------
# 2. Compute Image Gradients
# ----------------------------

# Horizontal gradient Ix
Ix = cv2.Sobel(
    gray,
    cv2.CV_32F,
    1, 0,
    ksize=3
)

# Vertical gradient Iy
Iy = cv2.Sobel(
    gray,
    cv2.CV_32F,
    0, 1,
    ksize=3
)


# ----------------------------
# 3. Construct Structure Tensor
# ----------------------------

Ix2 = Ix * Ix
Iy2 = Iy * Iy
Ixy = Ix * Iy


# Sum gradients in local neighbourhood
Sxx = cv2.GaussianBlur(Ix2, (5, 5), 1)
Syy = cv2.GaussianBlur(Iy2, (5, 5), 1)
Sxy = cv2.GaussianBlur(Ixy, (5, 5), 1)


# ----------------------------
# 4. Harris Equation
# ----------------------------

k = 0.04

# determinant:
# | Sxx  Sxy |
# | Sxy  Syy |
#
# det(M) = Sxx*Syy - Sxy^2

det_M = (Sxx * Syy) - (Sxy ** 2)

# trace(M) = Sxx + Syy
trace_M = Sxx + Syy

# Harris response
R = det_M - k * (trace_M ** 2)


# ----------------------------
# 5. Detect Strong Corners
# ----------------------------

threshold = 0.01 * R.max()

corner_locations = np.argwhere(R > threshold)

output = img.copy()


# Draw detected corners
for y, x in corner_locations:
    cv2.circle(
        output,
        (x, y),
        2,
        (0, 0, 255),
        -1
    )


# ----------------------------
# 6. Save Results
# ----------------------------

cv2.imwrite("harris_corners.jpg", output)

print("Corner detection finished!")
print("Number of corner pixels:", len(corner_locations))
print("Maximum Harris response:", R.max())
print("Result saved as harris_corners.jpg")
