import cv2
import numpy as np


image1 = cv2.imread("/data/projectwork/cv/Codex Image Sep 8, 2026, 11_17_23 PM.png", 0)
image2 = cv2.imread("/data/projectwork/cv/Codex Image Sep 8, 2026, 11_17_53 PM.png", 0)

if image1 is None or image2 is None:
    print("Could not open the images. Check the filenames.")
else:
   
    image1 = cv2.resize(image1, (512, 512))
    image2 = cv2.resize(image2, (512, 512))


    fft1 = np.fft.fftshift(np.fft.fft2(image1))
    fft2 = np.fft.fftshift(np.fft.fft2(image2))


    rows, cols = image1.shape
    y, x = np.ogrid[:rows, :cols]
    distance = (x - cols // 2) ** 2 + (y - rows // 2) ** 2
    sigma_low = 20
    sigma_high = 15
    low_filter = np.exp(-distance / (2 * sigma_low ** 2))
    high_filter = 1 - np.exp(-distance / (2 * sigma_high ** 2))
    low_part = fft1 * low_filter
    high_part = fft2 * high_filter
    combined = low_part + high_part
    result = np.fft.ifft2(np.fft.ifftshift(combined)).real
    result = np.clip(result, 0, 255).astype(np.uint8)
    cv2.imwrite("hybrid_result.png", result)
    print("Saved hybrid_result.png")
