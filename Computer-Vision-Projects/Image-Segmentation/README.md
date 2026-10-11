# Image Segmentation from Scratch

An image processing program that segments a photo three different ways,
using algorithms written from the ground up instead of a computer vision
library. Each pixel is treated as a node in a graph, and the three methods
analyze that structure differently:

| Method | What it does |
|---|---|
| **Felzenszwalb segmentation** | Sorts the edges between neighboring pixels by intensity difference and merges regions with a disjoint-set structure, using a size-sensitive threshold that keeps real boundaries while avoiding over-segmentation. |
| **SLIC superpixels** | Seeds a grid of cluster centers and iteratively assigns each pixel to the nearest center by a combined intensity and spatial distance, producing compact regions that follow edges. |
| **Laplacian of Gaussian blob detection** | Filters the image at a range of scales and finds local maxima in the scale-normalized response, marking blob-like features with circles sized to their scale. |

Their results can be overlaid to help a computer identify and separate the
objects in an image.

## How to run

```bash
pip install numpy matplotlib pillow scipy
```

| Command | What happens |
|---|---|
| `python app.py` | Opens a small window. Click **Upload Image & Segment**, pick a `.jpg`, `.png` or `.bmp`, and the four-panel result appears. |
| `python demo.py` | Runs all three methods on the sample photo in `images/` and shows the result. No file picker needed. |

`segmentation.py` holds the algorithms; `app.py` and `demo.py` only load an
image and call them.

## Data structures and complexity

The program relies on graphs, disjoint sets, lists, arrays, dictionaries and
stacks, chosen for time and memory efficiency in graph-based processing.

- **Felzenszwalb:** O(n log n), dominated by sorting the edges by weight.
- **SLIC:** O(n) per iteration, since each center only searches a local window.
- **Blob detection:** O(n × m) for n pixels across m scales.

## Limitations

- Image size is the main cost: the algorithms are plain Python, so large
  images take a long time and a lot of memory. Grayscale is processed, and
  `demo.py` shrinks the photo first for speed.
- Parameters are hard-coded and won't suit every image. Blurry images make
  SLIC and blob detection less accurate.
- The program visualizes segmentations but doesn't analyze or build on them.
  Telling overlapping or ambiguous objects apart (a person versus a statue,
  for example) needs further layers of analysis, such as deep learning models
  trained on labeled data.

## What I learned

The biggest challenges were learning Tkinter for the interface and
understanding how to represent and process an image as a graph. I wanted to
minimize library use, so much of the work was reverse-engineering how
standard libraries implement these methods. The result is less efficient
than those libraries, but it taught me how edges and nodes can be
manipulated, and how these ideas underpin applications like LIDAR mapping,
autonomous driving, medical imaging and satellite imagery analysis.

## Changes since the original version

- Split into `segmentation.py` (algorithms), `app.py` (window) and `demo.py`
  (sample run), so the algorithms can be reused and run without a window.
- **SLIC:** uses the standard distance from the SLIC paper: intensity on a
  0–100 scale, spatial distance scaled by the grid step, and distances reset
  each iteration. Default compactness is now 20. Before, intensity
  outweighed position so much that superpixels traced brightness instead of
  forming compact regions.
- **Blob detection:** intensities are scaled to 0–1 and the response is
  `sigma² × |LoG|`. Before, the threshold was applied to 0–255 values, so
  nearly every pixel registered as a blob.

## Photo credit

Sample photo by [Elijah Mears](https://unsplash.com/photos/CmKxr073k9k) on
Unsplash, converted to grayscale and resized.
