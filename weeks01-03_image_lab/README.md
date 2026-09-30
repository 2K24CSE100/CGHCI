 # Weeks 1–3 Image Processing Lab

## Photograph
The original photograph used for this lab is `images/original.jpg`. It contains photo of a wallet with a handwritten card reading “HCCGI Image Lab”.

## Task 1 — Image data
- Width: 1536 pixels
- Height: 1382 pixels
- Channels: 3
- NumPy shape: `[1382, 1536, 3]`
- Pixel count: 2,122,752
- Estimated raw image data at 8 bits/channel: 6,368,256 bytes
- Color order used by OpenCV: BGR

The image has 1536 columns and 1382 rows. Each pixel has three 8-bit color-channel values, so the raw uncompressed estimate is width × height × 3 bytes. JPEG file size is different because JPEG uses compression.

## Task 2 — Channels, grayscale, and downsampling
The script creates red, green, blue, grayscale, and half-size views in `outputs/pixel_views.png`.

- Original size: 1536 × 1382
- Downsampled size: 768 × 691
- One visible detail lost by downsampling: fine texture/detail becomes less distinct because fewer pixels represent the same scene.
- Channel difference: the red, green, and blue channel images have different brightness patterns because objects reflect the three color components differently.

## Task 3 — Adjustments
The script creates `outputs/adjustments.png` from the grayscale image.

- Brightness delta: +40
- Contrast factor: 1.5
- Threshold: 127
- Brightness and contrast are clipped to the valid 0–255 range.
- Threshold behavior: pixels greater than 127 become white (255); all other pixels become black (0).

## Task 4 — Blur and Sobel edges
The script creates `outputs/blur_and_edges.png`.

- Mean-blur kernel: 5 × 5
- Sobel edge calculation uses 3 × 3 Sobel kernels in the horizontal and vertical directions and combines their magnitudes.
- Visible comparison: strong boundaries such as the edge of the brown object against the lighter background remain prominent in the original edge image; after blurring, fine texture edges are reduced or softened.
- Detail reduced by blur: small texture lines and other fine/high-frequency details become less distinct.

## Installation
Install Python 3, then from the repository root run:

```bash
python3 -m pip install -r requirements.txt
```

## Exact run command

```bash
python3 weeks01-03_image_lab.py
```

The script reads `images/original.jpg` and creates/updates:
- `outputs/pixel_views.png`
- `outputs/adjustments.png`
- `outputs/blur_and_edges.png`

## Required repository structure

```text
images/original.jpg
weeks01-03_image_lab.py
outputs/pixel_views.png
outputs/adjustments.png
outputs/blur_and_edges.png
requirements.txt
README.md
```

## Notes
No pretrained model or online image-processing service is used. Processing is performed with OpenCV, NumPy, and Matplotlib.
