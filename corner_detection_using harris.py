import cv2
import numpy as np
img = cv2.imread("image.jpg")


gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

gray = np.float32(gray)

corners = cv2.cornerHarris(
    gray,
    blockSize=2,   
    ksize=3,       
    k=0.04        
)


corners = cv2.dilate(corners, None)


img[corners > 0.01 * corners.max()] = [0, 0, 255]

cv2.imshow("Harris Corners", img)

cv2.waitKey(0)
cv2.destroyAllWindows()
