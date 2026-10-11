# Jacque Fong
"""
demo.py — runs all three segmentation methods on a sample photo.

Same algorithms and settings as app.py, without the file picker. The photo
is shrunk first: the algorithms are written in plain Python (no vision
library), so run time grows quickly with the number of pixels.

Run:  python demo.py
"""

import time
from pathlib import Path

import numpy as np
from PIL import Image

from segmentation import felzenszwalb_segmentation, slic_grayscale, blob_log, plot_segmentation

IMAGE = Path(__file__).parent / "images" / "theatre.jpg"
DEMO_WIDTH = 300          # pixels; raise for more detail, lower for speed


def timed(label, func, *args, **kwargs):
    start = time.perf_counter()
    result = func(*args, **kwargs)
    print(f"{label:<28} {time.perf_counter() - start:6.1f} s")
    return result


img = Image.open(IMAGE).convert("L")
img = img.resize((DEMO_WIDTH, round(img.height * DEMO_WIDTH / img.width)), Image.LANCZOS)
image_np = np.array(img)
h, w = image_np.shape
print(f"Image: {IMAGE.name}, resized to {w} x {h} = {w * h:,} pixels (graph nodes)\n")

felz = timed("Felzenszwalb segmentation", felzenszwalb_segmentation, image_np, k=300)
slic = timed("SLIC superpixels", slic_grayscale, image_np, n_segments=200)
blobs = timed("LoG blob detection", blob_log, image_np, min_sigma=1, max_sigma=30, num_sigma=10, threshold=0.1)

print(f"\nFelzenszwalb regions: {len(np.unique(felz)):,}")
print(f"SLIC superpixels:     {len(np.unique(slic[slic >= 0])):,}")
print(f"Blobs detected:       {len(blobs):,}")

plot_segmentation(image_np, felz, slic, blobs)
