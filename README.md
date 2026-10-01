# Computer Vision from First Principles

**Explore how images become structure: gradients, corners, frequencies, and geometry.**

This is my growing collection of computer vision experiments in Python, NumPy, and OpenCV. The goal is to connect the mathematics to short implementations, inspect the intermediate steps, and build intuition before moving to larger vision systems.

> **Current focus:** classical vision foundations, including Harris corner detection, frequency-domain hybrid images, and orientation estimation with image moments.

## Explore the experiments

| Experiment | Implementation | Core idea |
| --- | --- | --- |
| Harris corners, step by step | [`harris_corner_detection.py`](harris_corner_detection.py) | Sobel gradients → structure tensor → Harris response → thresholding |
| OpenCV Harris comparison | [`corner_detection_using harris.py`](corner_detection_using%20harris.py) | Compare the explicit calculation with `cv2.cornerHarris` |
| Hybrid images | [`Hybrid_image.py`](Hybrid_image.py) | Combine low frequencies from one image with high frequencies from another |
| Object orientation | [`finding_orintetation.py`](finding_orintetation.py) | Use the centroid and second-order central moments to estimate a principal axis |

The scripts use NumPy and OpenCV operations while making the algorithm stages visible. Existing filenames are preserved so earlier links remain useful.

## Guided learning sequence

### 1. Understand local image structure

Start with [`harris_corner_detection.py`](harris_corner_detection.py). It computes horizontal and vertical gradients, smooths their products, and forms a local structure tensor:

```text
M = [[Sxx, Sxy],
     [Sxy, Syy]]

R = det(M) - k × trace(M)²
```

Large positive responses indicate candidate corner regions. Compare the response threshold with the OpenCV implementation. The current scripts mark thresholded corner pixels; non-maximum suppression is a planned refinement.

### 2. Inspect image frequencies

Read [`Hybrid_image.py`](Hybrid_image.py). A centered 2D Fourier transform separates low-frequency structure from high-frequency detail. Gaussian masks combine those components before an inverse transform reconstructs the hybrid image.

Try changing `sigma_low` and `sigma_high`, then inspect the result at different viewing sizes.

### 3. Estimate orientation from geometry

Read [`finding_orintetation.py`](finding_orintetation.py). After thresholding, the script treats dark pixels as foreground and estimates:

```text
theta = 0.5 × atan2(2 × u11, u20 - u02)
```

The principal-axis angle is defined modulo 180 degrees. Threshold choice, background clutter, and symmetric objects affect how meaningful the estimate is.

## Getting started

Use Python 3.10+:

```bash
git clone https://github.com/Vansh-A1/computer-Vision-from-first-principle-.git
cd computer-Vision-from-first-principle-
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows, activate with `.venv\Scripts\activate`.

### First experiment: Harris corners

Place your own image named `image.jpg` in the repository root, then run:

```bash
python harris_corner_detection.py
```

The script saves `harris_corners.jpg` and prints the number of selected corner pixels and maximum response.

### Other experiments

The other scripts currently contain example absolute image paths. Replace them with paths to your own inputs before running:

```bash
python "corner_detection_using harris.py"
python Hybrid_image.py
python finding_orintetation.py
```

For the OpenCV comparison, also update `output_path` to a writable location. Hybrid images require two inputs and produce `hybrid_result.png`. Orientation estimation uses Matplotlib to display the axis and centroid; a desktop plotting environment is needed.

## Development roadmap

- [x] Make the Harris response calculation explicit.
- [x] Add an OpenCV Harris comparison.
- [x] Construct hybrid images in the frequency domain.
- [x] Estimate foreground orientation using image moments.
- [ ] Replace example image paths with command-line arguments.
- [ ] Add portable sample images and expected output galleries.
- [ ] Visualize gradients, response maps, and frequency masks.
- [ ] Add non-maximum suppression and corner comparisons.
- [ ] Extend the collection with edges, pyramids, and feature matching.
- [ ] Document controlled experiments on noise, scale, and rotation.

## Related work

Explore my [underwater enhancement and tracking pipeline](https://github.com/Vansh-A1/Under-water-image-enhancement-and-tracking-) and [thermal super-resolution experiments](https://github.com/Vansh-A1/Thermal-SuperResolution-8x) for applications of vision beyond these foundations.

See [CONTRIBUTING.md](CONTRIBUTING.md) for focused improvements and experiment reporting.

Built and maintained by [Vansh Joshi](https://github.com/Vansh-A1).

