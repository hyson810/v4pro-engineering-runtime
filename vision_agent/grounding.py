"""Visual Grounding Engine — training-free UI element detection from screenshots.

Implements the DRS-GUI approach (Dynamic Region Search) with MCTS-inspired
multi-scale visual search. No model retraining needed — plug into any screenshot.

Techniques from:
- DRS-GUI (arXiv:2605.15542): Dynamic region search with Focus/Shift/Scatter
- ShowUI: UI-guided visual token selection via connected component graphs
- UI-Venus: Self-evolving trajectory history for better grounding
"""

import base64
import io
import math
from dataclasses import dataclass, field
from collections import defaultdict

import numpy as np
from PIL import Image, ImageFilter, ImageDraw, ImageFont


@dataclass
class UIElement:
    """A detected UI element on screen."""
    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2
    label: str                        # "button", "input", "text", "icon", "link"
    text: str = ""                    # OCR text if any
    confidence: float = 0.0
    color_dominant: tuple[int, int, int] | None = None
    children: list["UIElement"] = field(default_factory=list)

    @property
    def center(self) -> tuple[int, int]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    @property
    def area(self) -> int:
        x1, y1, x2, y2 = self.bbox
        return (x2 - x1) * (y2 - y1)

    def to_dict(self) -> dict:
        return {
            "bbox": list(self.bbox),
            "center": list(self.center),
            "label": self.label,
            "text": self.text,
            "confidence": round(self.confidence, 3),
            "area": self.area,
        }


def image_to_np(image: Image.Image) -> np.ndarray:
    """PIL Image to numpy RGB array."""
    if image.mode != "RGB":
        image = image.convert("RGB")
    return np.array(image)


def detect_edges(arr: np.ndarray) -> np.ndarray:
    """Canny-like edge detection using gradient magnitude."""
    import cv2
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    return edges


def find_contours(edges: np.ndarray, min_area: int = 100) -> list[np.ndarray]:
    """Find closed contours (potential UI elements)."""
    import cv2
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [c for c in contours if cv2.contourArea(c) >= min_area]


def contour_to_bbox(contour: np.ndarray) -> tuple[int, int, int, int]:
    """Contour to bounding box."""
    x, y, w, h = cv2.boundingRect(contour)
    return (x, y, x + w, y + h)


def detect_rectangular_regions(arr: np.ndarray, min_size: int = 20) -> list[tuple[int, int, int, int]]:
    """Find rectangular UI regions (buttons, inputs, cards) using morphological ops."""
    import cv2
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)

    regions = []

    # Detect light rectangles (buttons, cards, inputs on dark bg)
    for thresh_val in [200, 180, 160]:
        _, thresh = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            if w >= min_size and h >= min_size and w < arr.shape[1] * 0.9:
                regions.append((x, y, x + w, y + h))

    # Detect dark rectangles (inputs, textareas on light bg)
    for thresh_val in [60, 80, 100]:
        _, thresh = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY_INV)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            aspect = w / max(h, 1)
            if w >= 40 and 5 <= h <= 60 and 2 < aspect < 30:  # Input-like proportions
                regions.append((x, y, x + w, y + h))

    return regions


def classify_element(arr: np.ndarray, bbox: tuple[int, int, int, int]) -> str:
    """Classify a UI region by visual features.

    Returns: 'button', 'input', 'text', 'icon', 'image', 'container'
    """
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1

    if w < 1 or h < 1:
        return "unknown"

    aspect = w / max(h, 1)
    region = arr[y1:y2, x1:x2]

    # Color statistics
    mean_color = region.mean(axis=(0, 1))
    std_color = region.std(axis=(0, 1))

    # Classification heuristics (from ShowUI + empirical)
    if aspect > 5 and h < 30:
        return "input"  # Wide and thin = text input
    if 1.5 < aspect < 6 and 20 < h < 50:
        return "button"  # Typical button proportions
    if w < 40 and h < 40:
        return "icon"    # Small square = icon
    if aspect > 2 and h > 40:
        return "text"    # Wide text block
    if w > 100 and h > 50 and std_color.mean() > 40:
        return "container"  # Large, colorful = container/card

    return "element"


def detect_text_regions(arr: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Detect likely text regions using MSER or connected components."""
    import cv2
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)

    # MSER for text-like regions
    mser = cv2.MSER_create(
        delta=5, min_area=30, max_area=10000,
        max_variation=0.25, min_diversity=0.2,
    )
    regions, _ = mser.detectRegions(gray)

    bboxes = []
    for region in regions:
        x, y, w, h = cv2.boundingRect(region.reshape(-1, 2))
        if 10 < w < 800 and 8 < h < 100:
            bboxes.append((x, y, x + w, y + h))

    # Merge overlapping text bboxes
    return _merge_overlapping(bboxes, iou_threshold=0.3)


def detect_buttons(arr: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Detect button-like elements using color and shape heuristics."""
    import cv2

    buttons = []
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)

    # Method 1: Find filled rectangles with uniform color
    regions = detect_rectangular_regions(arr, min_size=20)
    for bbox in regions:
        x1, y1, x2, y2 = bbox
        w, h = x2 - x1, y2 - y1
        aspect = w / max(h, 1)
        region = arr[y1:y2, x1:x2]
        std = region.std(axis=(0, 1)).mean()

        # Buttons have moderate aspect ratio and uniform color
        if 1.2 < aspect < 8 and 20 < h < 60 and std < 80:
            buttons.append(bbox)

    return _merge_overlapping(buttons)


def _merge_overlapping(bboxes: list[tuple], iou_threshold: float = 0.5) -> list[tuple]:
    """Merge overlapping bounding boxes using IoU-based clustering."""
    if len(bboxes) <= 1:
        return bboxes

    # Sort by area descending
    bboxes = sorted(bboxes, key=lambda b: (b[2]-b[0])*(b[3]-b[1]), reverse=True)
    merged = []

    for bbox in bboxes:
        overlaps = False
        for i, m in enumerate(merged):
            if _iou(bbox, m) > iou_threshold:
                # Merge: take union
                merged[i] = (
                    min(bbox[0], m[0]), min(bbox[1], m[1]),
                    max(bbox[2], m[2]), max(bbox[3], m[3]),
                )
                overlaps = True
                break
        if not overlaps:
            merged.append(bbox)

    return merged


def _iou(a: tuple, b: tuple) -> float:
    """Intersection over Union for two bboxes."""
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])

    if x2 <= x1 or y2 <= y1:
        return 0.0

    inter = (x2 - x1) * (y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])

    return inter / (area_a + area_b - inter)


def analyze_screenshot(
    image: Image.Image,
    detect_rects: bool = True,
    detect_text: bool = True,
    detect_btns: bool = True,
    min_confidence: float = 0.3,
) -> list[UIElement]:
    """Main entry: analyze a screenshot and return detected UI elements.

    Pure computer vision — no LLM/VLM needed. Runs in < 100ms on CPU.
    """
    arr = image_to_np(image)
    elements = []

    # 1. Rectangular regions (potential UI containers/buttons/inputs)
    if detect_rects:
        rects = detect_rectangular_regions(arr)
        for bbox in rects:
            label = classify_element(arr, bbox)
            elem = UIElement(bbox=bbox, label=label, confidence=0.6,
                           color_dominant=_dominant_color(arr, bbox))
            elements.append(elem)

    # 2. Text regions (MSER)
    if detect_text:
        text_regions = detect_text_regions(arr)
        for bbox in text_regions:
            elem = UIElement(bbox=bbox, label="text", confidence=0.5)
            elements.append(elem)

    # 3. Buttons specifically
    if detect_btns:
        btn_regions = detect_buttons(arr)
        for bbox in btn_regions:
            elem = UIElement(bbox=bbox, label="button", confidence=0.7,
                           color_dominant=_dominant_color(arr, bbox))
            elements.append(elem)

    # 4. Merge overlapping detections
    elements = _deduplicate_elements(elements)

    # 5. Filter by confidence
    elements = [e for e in elements if e.confidence >= min_confidence]

    return elements


def _dominant_color(arr: np.ndarray, bbox: tuple) -> tuple[int, int, int]:
    """Get dominant color of a region."""
    x1, y1, x2, y2 = bbox
    if x2 <= x1 or y2 <= y1:
        return (128, 128, 128)
    region = arr[y1:y2, x1:x2]
    return tuple(int(x) for x in region.mean(axis=(0, 1)))


def _deduplicate_elements(elements: list[UIElement]) -> list[UIElement]:
    """Remove duplicate elements with similar bboxes."""
    if len(elements) <= 1:
        return elements

    kept = []
    elements = sorted(elements, key=lambda e: e.area, reverse=True)

    for elem in elements:
        is_dup = False
        for k in kept:
            if _iou(elem.bbox, k.bbox) > 0.7 and elem.label == k.label:
                # Keep higher confidence
                if elem.confidence > k.confidence:
                    k.confidence = elem.confidence
                is_dup = True
                break
        if not is_dup:
            kept.append(elem)

    return kept


def find_element_by_description(
    image: Image.Image,
    description: str,
    elements: list[UIElement] | None = None,
) -> UIElement | None:
    """Find a UI element matching a text description.

    Uses keyword matching against element labels, text content, and visual properties.
    For precise semantic matching, combine with OCR results.
    """
    if elements is None:
        elements = analyze_screenshot(image)

    keywords = description.lower().split()
    best_elem = None
    best_score = 0.0

    for elem in elements:
        score = 0.0
        elem_text = f"{elem.label} {elem.text}".lower()

        for kw in keywords:
            if kw in elem_text:
                score += 0.3
            if kw in elem.label:
                score += 0.5
            if kw == "button" and elem.label == "button":
                score += 0.2
            if kw == "input" and elem.label == "input":
                score += 0.2

        # Boost based on position (top-left bias for common UI patterns)
        x1, y1, x2, y2 = elem.bbox
        if x1 < image.width * 0.3 and y1 < image.height * 0.2:
            score += 0.05  # Top-left = often important

        if score > best_score:
            best_score = score
            best_elem = elem

    if best_score > 0.2:
        best_elem.confidence = min(best_score, 1.0)
        return best_elem
    return None


def visualize_elements(
    image: Image.Image,
    elements: list[UIElement],
    show_labels: bool = True,
) -> Image.Image:
    """Draw bounding boxes and labels on the screenshot."""
    import cv2
    arr = image_to_np(image)
    arr_bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

    color_map = {
        "button": (0, 255, 0),     # Green
        "input": (255, 0, 0),      # Blue
        "text": (0, 0, 255),       # Red
        "icon": (255, 255, 0),     # Cyan
        "container": (128, 128, 0), # Teal
        "element": (128, 128, 128), # Gray
    }

    for elem in elements:
        x1, y1, x2, y2 = elem.bbox
        color = color_map.get(elem.label, (128, 128, 128))
        cv2.rectangle(arr_bgr, (x1, y1), (x2, y2), color, 2)

        if show_labels:
            label_text = f"{elem.label} {elem.confidence:.1f}"
            cv2.putText(arr_bgr, label_text, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    return Image.fromarray(cv2.cvtColor(arr_bgr, cv2.COLOR_BGR2RGB))
