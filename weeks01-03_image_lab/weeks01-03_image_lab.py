import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


def inspect_image(image_path: str) -> dict:
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    height, width, channels = image.shape
    return {
        "width": width,
        "height": height,
        "channels": channels,
        "shape": list(image.shape),
        "pixel_count": width * height,
        "estimated_bytes": width * height * channels,
        "color_order": "BGR",
    }


def _save_montage(images, titles, output_path, cols=3):
    rows = int(np.ceil(len(images) / cols))
    plt.figure(figsize=(12, 4 * rows))
    for i, (img, title) in enumerate(zip(images, titles), start=1):
        ax = plt.subplot(rows, cols, i)
        if img.ndim == 2:
            ax.imshow(img, cmap="gray")
        else:
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title(title)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def create_pixel_views(image_path: str, output_dir: str) -> dict:
    os.makedirs(output_dir, exist_ok=True)
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    b, g, r = cv2.split(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = image.shape[:2]
    new_width, new_height = width // 2, height // 2
    downsampled = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    output_path = os.path.join(output_dir, "pixel_views.png")
    _save_montage(
        [r, g, b, gray, downsampled],
        ["Red channel", "Green channel", "Blue channel", "Grayscale",
         f"Half-size ({new_width} x {new_height})"],
        output_path, cols=3)
    return {"original_size": [width, height], "downsampled_size": [new_width, new_height], "output_path": output_path}


def create_adjustments(image_path: str, output_dir: str, brightness_delta: int = 40,
                       contrast_factor: float = 1.5, threshold: int = 127) -> dict:
    os.makedirs(output_dir, exist_ok=True)
    if not 0 <= threshold <= 255:
        raise ValueError("threshold must be in the range 0-255")
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    brighter = np.clip(gray.astype(np.float32) + brightness_delta, 0, 255).astype(np.uint8)
    higher_contrast = np.clip(gray.astype(np.float32) * contrast_factor, 0, 255).astype(np.uint8)
    binary = np.where(gray > threshold, 255, 0).astype(np.uint8)
    output_path = os.path.join(output_dir, "adjustments.png")
    _save_montage(
        [gray, brighter, higher_contrast, binary],
        ["Original grayscale", f"Brighter (+{brightness_delta})",
         f"Higher contrast (x{contrast_factor})", f"Threshold ({threshold})"],
        output_path, cols=2)
    return {"brightness_delta": brightness_delta, "contrast_factor": contrast_factor,
            "threshold": threshold, "output_path": output_path}


def create_blur_and_edges(image_path: str, output_dir: str, kernel_size: int = 5) -> dict:
    os.makedirs(output_dir, exist_ok=True)
    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be a positive odd integer")
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.blur(gray, (kernel_size, kernel_size))
    ox = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    oy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    bx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    by = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    original_edges = cv2.normalize(cv2.magnitude(ox, oy), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    blurred_edges = cv2.normalize(cv2.magnitude(bx, by), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    output_path = os.path.join(output_dir, "blur_and_edges.png")
    _save_montage(
        [gray, blurred, original_edges, blurred_edges],
        ["Grayscale", f"Mean blur ({kernel_size}x{kernel_size})",
         "Sobel edges — original", "Sobel edges — blurred"],
        output_path, cols=2)
    return {"kernel_size": kernel_size, "output_path": output_path}


def run_lab(image_path: str, output_dir: str) -> dict:
    return {
        "task1": inspect_image(image_path),
        "task2": create_pixel_views(image_path, output_dir),
        "task3": create_adjustments(image_path, output_dir),
        "task4": create_blur_and_edges(image_path, output_dir),
    }


def main() -> None:
    results = run_lab("images/original.jpg", "outputs")
    print("Task 1:", results["task1"])
    print("Task 2:", results["task2"])
    print("Task 3:", results["task3"])
    print("Task 4:", results["task4"])


if __name__ == "__main__":
    main()
