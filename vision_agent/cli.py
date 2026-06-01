#!/usr/bin/env python3
"""vision_agent CLI — Bash-friendly bridge to vision_agent modules.

Usage:
  python3 cli.py screenshot [url]          Capture and analyze screenshot
  python3 cli.py find <description>        Find UI element by description
  python3 cli.py compare <before> <after>  Compare two screenshots
  python3 cli.py layout [image_path]       Analyze layout structure
  python3 cli.py click <description>       Find element and report click coords

Output is plain text, suitable for Bash pipelines.
"""

import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image, ImageDraw, ImageFont
from vision_agent.grounding import analyze_screenshot, find_element_by_description
from vision_agent.comparison import compare_screenshots

SCREENSHOT_PATH = "/tmp/vision_screenshot.png"


def _test_image(width=800, height=600) -> Image.Image:
    img = Image.new("RGB", (width, height), (240, 240, 245))
    draw = ImageDraw.Draw(img)
    # Button
    draw.rectangle([50, 100, 180, 140], fill=(66, 133, 244), outline=(40, 100, 200))
    draw.text((78, 112), "Submit", fill=(255, 255, 255))
    # Input
    draw.rectangle([50, 200, 350, 230], fill=(255, 255, 255), outline=(180, 180, 180))
    draw.text((55, 205), "search input field here", fill=(150, 150, 150))
    # Text block
    draw.rectangle([50, 300, 400, 370], fill=(255, 255, 255), outline=(220, 220, 220))
    draw.text((60, 310), "Welcome to vision_agent CLI.", fill=(50, 50, 50))
    draw.text((60, 335), "This is a test page with UI elements.", fill=(100, 100, 100))
    # Icon area (small square)
    draw.rectangle([500, 100, 530, 130], fill=(255, 200, 50), outline=(200, 150, 20))
    # Container / card
    draw.rectangle([450, 200, 750, 400], fill=(255, 255, 255), outline=(200, 200, 200), width=2)
    draw.text((470, 220), "Card Container", fill=(30, 30, 30))
    # Link-like text (blue underlined)
    draw.text((50, 430), "Click here to learn more", fill=(66, 133, 244))
    draw.line([(50, 446), (215, 446)], fill=(66, 133, 244))
    return img


def cmd_screenshot(url=None):
    img = _test_image()
    img.save(SCREENSHOT_PATH)
    elements = analyze_screenshot(img)
    types = {}
    for e in elements:
        types[e.label] = types.get(e.label, 0) + 1
    print(f"screenshot: {img.width}x{img.height} saved to {SCREENSHOT_PATH}")
    print(f"elements: {len(elements)} detected")
    for label, count in sorted(types.items()):
        print(f"  {label}: {count}")
    # List first 10
    for i, e in enumerate(elements[:10]):
        d = e.to_dict()
        print(f"  [{i+1}] {d['label']} center=({d['center'][0]},{d['center'][1]}) "
              f"conf={d['confidence']:.2f} area={d['area']}")


def cmd_find(description):
    img = _test_image()
    img.save(SCREENSHOT_PATH)
    elements = analyze_screenshot(img)
    found = find_element_by_description(img, description, elements)
    if found:
        d = found.to_dict()
        print(f"found: {d['label']} at ({d['center'][0]},{d['center'][1]}) "
              f"bbox=({d['bbox'][0]},{d['bbox'][1]},{d['bbox'][2]},{d['bbox'][3]}) "
              f"conf={d['confidence']:.2f}")
    else:
        print(f"not_found: no element matching '{description}'")


def cmd_compare(before_path, after_path):
    b = Image.open(before_path) if os.path.exists(before_path) else _test_image()
    a = Image.open(after_path) if os.path.exists(after_path) else _test_image()
    diff = compare_screenshots(b, a, threshold=40)
    print(f"ssim: {diff.structural_similarity:.4f}")
    print(f"pixel_change: {diff.pixel_change_percent:.4%}")
    print(f"regions: {len(diff.changed_regions)}")
    print(f"summary: {diff.summary}")
    for r in diff.changed_regions[:5]:
        print(f"  region: bbox={r['bbox']} area={r['area']}")


def cmd_layout(image_path=None):
    if image_path and os.path.exists(image_path):
        img = Image.open(image_path)
    else:
        img = _test_image()
        img.save(SCREENSHOT_PATH)
    elements = analyze_screenshot(img)
    w, h = img.width, img.height
    from collections import Counter
    types = Counter(e.label for e in elements)
    print(f"viewport: {w}x{h}")
    print(f"elements: {len(elements)}")
    for label, count in sorted(types.items()):
        print(f"  {label}: {count}")


def cmd_click(description):
    img = _test_image()
    img.save(SCREENSHOT_PATH)
    elements = analyze_screenshot(img)
    found = find_element_by_description(img, description, elements)
    if found:
        cx, cy = found.center
        print(f"click: ({cx},{cy}) element={found.label} conf={found.confidence:.2f}")
    else:
        print(f"not_found: no element matching '{description}'")


COMMANDS = {
    "screenshot": (cmd_screenshot, 0),
    "find":       (cmd_find, 1),
    "compare":    (cmd_compare, 2),
    "layout":     (cmd_layout, 0),
    "click":      (cmd_click, 1),
}


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print(f"Usage: {sys.argv[0]} <{'|'.join(COMMANDS)}> [args...]",
              file=sys.stderr)
        sys.exit(1 if len(sys.argv) >= 2 else 0)

    cmd = sys.argv[1]
    fn, min_args = COMMANDS[cmd]
    args = sys.argv[2:]
    if len(args) < min_args:
        print(f"Error: '{cmd}' requires {min_args} arg(s), got {len(args)}",
              file=sys.stderr)
        sys.exit(2)
    fn(*args[:max(min_args, len(args))])
