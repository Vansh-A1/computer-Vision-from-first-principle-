```python
import cv2
import numpy as np
import matplotlib.pyplot as plt


def get_orientation(image_path):

    image = cv2.imread(image_path)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    _, thresh = cv2.threshold(gray, 170, 255, cv2.THRESH_BINARY)

    y, x = np.where(thresh == 0)

    x_mean = np.mean(x)
    y_mean = np.mean(y)

    x = x - x_mean
    y = y - y_mean

    u20 = np.sum(x**2)
    u02 = np.sum(y**2)
    u11 = np.sum(x*y)

    theta = 0.5 * np.arctan2(2*u11, u20-u02)

    print("Orientation:", theta, "radians")
    print("Orientation:", np.degrees(theta), "degrees")
    print("Centroid:", x_mean, y_mean)

    return theta, x_mean, y_mean
def plot_orientation(image_path):

    theta, x_mean, y_mean = get_orientation(image_path)

    image = cv2.imread(image_path)

    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    x_line = np.linspace(0, image.shape[1], 500)

    y_line = np.tan(theta) * (x_line - x_mean) + y_mean

    plt.plot(x_line, y_line, 'r', linewidth=2)

    plt.scatter(x_mean, y_mean, color='blue', s=50)

    plt.title("Orientation = {:.2f}°".format(np.degrees(theta)))

    plt.axis("equal")
    plt.show()


plot_orientation("/data/projectwork/cv/images.jpeg")
```
