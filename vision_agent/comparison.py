"""Visual Comparison Engine — fast screenshot diff and change detection.

Techniques:
- Structural Similarity Index (SSIM) for perceptual diff
- Pixel-level delta with thresholding
- Layout change detection
- Element-level before/after tracking
"""

import io
import base64
import math
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageChops, ImageStat


@dataclass
class DiffResult:
    """Result of comparing two screenshots."""
    changed: bool
    structural_similarity: float      # 0-1, 1 = identical
    pixel_change_percent: float       # % of pixels that changed
    changed_regions: list[dict]        # bounding boxes of changed areas
    summary: str


def image_to_np(image: Image.Image) -> np.ndarray:
    if image.mode != "RGB":
        image = image.convert("RGB")
    return np.array(image)


def structural_similarity(img1: Image.Image, img2: Image.Image) -> float:
    """Fast SSIM approximation using local statistics."""
    arr1 = image_to_np(img1).astype(np.float64)
    arr2 = image_to_np(img2).astype(np.float64)

    # Resize to same dimensions if needed
    if arr1.shape != arr2.shape:
        h = min(arr1.shape[0], arr2.shape[0])
        w = min(arr1.shape[1], arr2.shape[1])
        arr1 = arr1[:h, :w]
        arr2 = arr2[:h, :w]

    # Simplified SSIM: mean of per-pixel normalized similarity
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2

    mu1 = arr1.mean(axis=2)
    mu2 = arr2.mean(axis=2)
    sigma1_sq = arr1.var(axis=2)
    sigma2_sq = arr2.var(axis=2)
    sigma12 = ((arr1 - mu1[..., None]) * (arr2 - mu2[..., None])).mean(axis=2)

    numerator = (2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)
    denominator = (mu1**2 + mu2**2 + c1) * (sigma1_sq + sigma2_sq + c2)

    ssim_map = numerator / (denominator + 1e-10)
    return float(np.clip(ssim_map.mean(), 0.0, 1.0))


def pixel_diff(img1: Image.Image, img2: Image.Image, threshold: int = 30) -> float:
    """Percentage of pixels that changed beyond threshold."""
    arr1 = image_to_np(img1).astype(np.int16)
    arr2 = image_to_np(img2).astype(np.int16)

    if arr1.shape != arr2.shape:
        h = min(arr1.shape[0], arr2.shape[0])
        w = min(arr1.shape[1], arr2.shape[1])
        arr1 = arr1[:h, :w]
        arr2 = arr2[:h, :w]

    diff = np.abs(arr1 - arr2).max(axis=2)
    changed = (diff > threshold).sum()
    total = diff.size

    return float(changed / total)


def find_changed_regions(
    img1: Image.Image,
    img2: Image.Image,
    threshold: int = 40,
    min_area: int = 50,
) -> list[dict]:
    """Find bounding boxes of significantly changed regions."""
    import cv2

    arr1 = image_to_np(img1)
    arr2 = image_to_np(img2)

    if arr1.shape != arr2.shape:
        h = min(arr1.shape[0], arr2.shape[0])
        w = min(arr1.shape[1], arr2.shape[1])
        arr1 = arr1[:h, :w]
        arr2 = arr2[:h, :w]

    # Compute per-pixel difference
    diff = cv2.absdiff(arr1, arr2)
    gray = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

    # Dilate to connect nearby changes
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilated = cv2.dilate(thresh, kernel, iterations=2)

    # Find contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    regions = []
    for c in contours:
        area = cv2.contourArea(c)
        if area >= min_area:
            x, y, w, h = cv2.boundingRect(c)
            regions.append({
                "bbox": [x, y, x + w, y + h],
                "area": int(area),
                "center": [x + w // 2, y + h // 2],
            })

    return regions


def highlight_diff(
    img1: Image.Image,
    img2: Image.Image,
    threshold: int = 40,
) -> Image.Image:
    """Create a visualization of the diff between two screenshots."""
    import cv2

    arr1 = image_to_np(img1)
    arr2 = image_to_np(img2)

    if arr1.shape != arr2.shape:
        h = min(arr1.shape[0], arr2.shape[0])
        w = min(arr1.shape[1], arr2.shape[1])
        arr1 = arr1[:h, :w]
        arr2 = arr2[:h, :w]

    diff = cv2.absdiff(arr1, arr2)
    gray = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

    # Overlay diff in red on img2
    result = arr2.copy()
    result[thresh > 0] = [255, 0, 0]  # Red overlay

    # Blend 50% with original
    alpha = 0.5
    blended = (alpha * arr2 + (1 - alpha) * result).astype(np.uint8)

    return Image.fromarray(blended)


def compare_screenshots(
    before: Image.Image,
    after: Image.Image,
    threshold: int = 40,
) -> DiffResult:
    """Full comparison pipeline."""
    ssim = structural_similarity(before, after)
    px_change = pixel_diff(before, after, threshold=threshold)
    regions = find_changed_regions(before, after, threshold=threshold)

    changed = px_change > 0.001  # More than 0.1% changed

    if not changed:
        summary = "No visual changes detected."
    elif px_change < 0.01:
        summary = f"Minor changes: {px_change:.2%} of pixels, {len(regions)} region(s)."
    elif px_change < 0.1:
        summary = f"Moderate changes: {px_change:.2%} of pixels, {len(regions)} region(s)."
    else:
        summary = f"Major changes: {px_change:.2%} of pixels, {len(regions)} region(s)."

    return DiffResult(
        changed=changed,
        structural_similarity=round(ssim, 4),
        pixel_change_percent=round(px_change, 4),
        changed_regions=regions[:10],  # Top 10
        summary=summary,
    )


def image_to_base64(image: Image.Image, format: str = "PNG") -> str:
    """Encode PIL Image as base64 data URL."""
    buf = io.BytesIO()
    image.save(buf, format=format)
    return base64.b64encode(buf.getvalue()).decode()
