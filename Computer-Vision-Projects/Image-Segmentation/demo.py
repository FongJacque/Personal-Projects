# Jacque Fong
"""
demo.py — runs all three segmentation methods on a photo.

Same algorithms and settings as app.py, without the file picker. Uses the
sample photo in images/, or your own image if you pass one:

    python demo.py
    python demo.py path/to/your_photo.jpg

The image is shrunk first: the algorithms are written in plain Python (no
vision library), so run time grows quickly with the number of pixels.
"""

import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from segmentation import felzenszwalb_segmentation, slic_grayscale, blob_log, plot_segmentation

SAMPLE = Path(__file__).parent / "images" / "theatre.jpg"
DEMO_SIZE = 300           # longest side in pixels; raise for more detail, lower for speed
MIN_SIZE = 40             # smaller than this leaves too few pixels for 200 superpixels


def timed(label, func, *args, **kwargs):
    start = time.perf_counter()
    result = func(*args, **kwargs)
    print(f"{label:<28} {time.perf_counter() - start:6.1f} s")
    return result


def load_grayscale(path):
    """Opens an image, applies the phone camera's rotation tag, converts to grayscale and shrinks it."""
    try:
        img = Image.open(path)
        img = ImageOps.exif_transpose(img)        # phone photos are often stored sideways with a rotation tag
    except (UnidentifiedImageError, OSError):
        sys.exit(f"Couldn't read {Path(path).name} as an image. Try a JPG, PNG, BMP or WebP file.")
    img = img.convert("L")
    scale = DEMO_SIZE / max(img.size)
    if scale < 1:
        img = img.resize((round(img.width * scale), round(img.height * scale)), Image.LANCZOS)
    if min(img.size) < MIN_SIZE:
        sys.exit(f"That image is too small ({img.width} x {img.height}). Use one at least {MIN_SIZE} pixels on each side.")
    return np.array(img)


path = Path(sys.argv[1]) if len(sys.argv) > 1 else SAMPLE
image_np = load_grayscale(path)
h, w = image_np.shape
source = "your image" if path != SAMPLE else f"sample photo ({SAMPLE.name})"
print(f"Using {source}, resized to {w} x {h} = {w * h:,} pixels (graph nodes)\n")

felz = timed("Felzenszwalb segmentation", felzenszwalb_segmentation, image_np, k=300)
slic = timed("SLIC superpixels", slic_grayscale, image_np, n_segments=200)
blobs = timed("LoG blob detection", blob_log, image_np, min_sigma=1, max_sigma=30, num_sigma=10, threshold=0.1)

print(f"\nFelzenszwalb regions: {len(np.unique(felz)):,}")
print(f"SLIC superpixels:     {len(np.unique(slic[slic >= 0])):,}")
print(f"Blobs detected:       {len(blobs):,}")

plot_segmentation(image_np, felz, slic, blobs)
