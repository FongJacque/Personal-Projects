# Jacque Fong
"""
segmentation.py — from-scratch image segmentation and feature detection.

Three algorithms, implemented without a computer-vision library:
  * Felzenszwalb graph-based segmentation (disjoint sets over a pixel graph)
  * SLIC superpixels (iterative clustering on intensity + position)
  * Laplacian of Gaussian (LoG) blob detection

app.py wraps these in a small desktop window; demo.py runs them on a
sample photo.
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from collections import namedtuple
import scipy.ndimage as ndi

Edge = namedtuple("Edge", ["a", "b", "weight"])

class DisjointSet:
    def __init__(self, size):
        """
        Initializes individual sets for graphical comparison.

        Parameters:
        None

        Returns:
        None

        Learning Notes (line by line):
        Creates a list of the size of the image (making separate disjointed sets for each pixel)
        Sets size of each individually created set to 1
        Stores internal differences for merging with graph-based image segmentation
            Compares the connections to adjacent nodes to determine segments or regions
        """
        self.parent = list(range(size))
        self.size = [1] * size
        self.int_diff = [0] * size

    def find(self, x):
        """
        Finds the root of the set containing x.

        Parameters:
        x (int): The element to find the root of.

        Returns:
        self.parent[x] (int): The root of the set containing x.

        Learning Notes (line by line):
        Recursively finds the root of the set containing x
            Uses path compression to optimize the recursion
        """
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])

        return self.parent[x]

    def union(self, x, y, weight, threshold):
        """
        Merges two sets if the weight is less than the threshold.

        Parameters:
        x (int): The first element to merge.
        y (int): The second element to merge.
        weight (float): The weight of the edge between x and y.
        threshold (float): The threshold for merging.

        Returns:
        False (bool): If the sets are already connected.
        True (bool): If the sets were successfully merged.

        Learning Notes (line by line):
        Initiliazes the roots of the two sets to be merged
        Checks to see if both pixels are already in the same set, if so, return False as no merge necessary
        Calculates the internal differences of the two sets
            Internal differences are used to determine the threshold for merging
            Merge condition is size-sensitive, meaning larger sets can be merged with smaller sets
            Balances detail and prevents over-segmentation
        If edge connecting the two sets is less than the threshold, merge the two sets
            Preserves boudaries between signficiantly different regions
                Used in Felzenszwalb algorithm
        Ensures the larger set becomes the parent of the smaller set
        Merges x and y by setting the parent of y to x, then updates the size and internal difference of x
        """
        x_root = self.find(x)
        y_root = self.find(y)
        if x_root == y_root:
            return False
        
        diff_x = self.int_diff[x_root] + threshold / self.size[x_root]
        diff_y = self.int_diff[y_root] + threshold / self.size[y_root]

        if weight > min(diff_x, diff_y):
            return False
        
        if self.size[x_root] < self.size[y_root]:
            x_root, y_root = y_root, x_root

        self.parent[y_root] = x_root
        self.size[x_root] += self.size[y_root]
        self.int_diff[x_root] = max(self.int_diff[x_root], self.int_diff[y_root], weight)
        return True


def build_graph(image):
    """
    Builds a graph from an image where each pixel is a node and edges connect neighboring pixels.

    Parameters:
    image (np.ndarray): A 2D grayscale image represented as a NumPy array.

    Returns:
    edges (list of Edge): A list of edges connecting neighboring pixels, each with a weight.
    h (int): The height of the image.
    w (int): The width of the image.

    Learning Notes (line by line):
    Retrieves the height and width of the image
    Initializes an empty list to store the edges
    Defines a lambda function to convert 2D (y, x) coordinates to a 1D index
        Represents the image as a flat graph structure
    Iterates over each pixel in the image
        Computes the 1D index for the current pixel
        For each pixel, checks two neighbors: right (0, 1) and bottom (1, 0)
            Ensures neighbors are within image boundaries
            Calculates the 1D index of the neighbor
            Computes the weight as the absolute difference in intensity between the pixel and its neighbor
            Creates an Edge between the current pixel and the neighbor with the computed weight
            Appends the Edge to the list
    Returns the complete list of edges and the image dimensions
    """
    h, w = image.shape
    edges = []
    index = lambda y, x: y * w + x
    for y in range(h):
        for x in range(w):
            current = index(y, x)
            for dy, dx in [(0, 1), (1, 0)]:
                ny, nx = y + dy, x + dx
                if ny < h and nx < w:
                    neighbor = index(ny, nx)
                    weight = abs(int(image[y, x]) - int(image[ny, nx]))
                    edges.append(Edge(current, neighbor, weight))
    return edges, h, w

def felzenszwalb_segmentation(image, k=300):
    """
    Segments an image using the Felzenszwalb segmentation algorithm.
        Felzenszwalb is a graph-based segmentation algorithm that merges regions based on edge weights and internal differences.

    Parameters:
    image (np.ndarray): A 2D grayscale image represented as a NumPy array.
    k (int, optional): The threshold value for segmentation. Default is 300.

    Returns:
    np.ndarray: A 2D array where each pixel is labeled with an integer representing the segment it belongs to.

    Learning Notes (line by line):
    Builds a graph from the image using the build_graph function
    Creates a DisjointSet Object for each node in the graph
    Checks each edge in the graph and merges the sets if the weight is less than k
    AFTER checking every node, outputs a 2D array where each pixel is labeled with the segment it belongs to, reassigned back into a 2D array
        Limits each array to a 32-bit signed integer to save memory
    Iterates over each pixel in the image and assigns the segment label to its segment
    """
    edges, h, w = build_graph(image)
    ds = DisjointSet(h * w)
    edges.sort(key=lambda e: e.weight)
    for edge in edges:
        ds.union(edge.a, edge.b, edge.weight, k)
    output = np.zeros((h, w), dtype=np.int32)
    for y in range(h):
        for x in range(w):
            output[y, x] = ds.find(y * w + x)
    return output

def slic_grayscale(image, n_segments=200, compactness=20, max_iter=10):
    """
    Segments a grayscale image using the SLIC (Simple Linear Iterative Clustering) algorithm.
        SLIC is a region-based segmentation algorithm that groups pixels into superpixels based on spatial proximity and color similarity.

    Parameters:
    image (np.ndarray): A 2D grayscale image represented as a NumPy array.
    n_segments (int, optional): The number of superpixels to generate. Default is 200.
    compactness (float, optional): A weight that balances the color distance and spatial distance. Default is 20.
    max_iter (int, optional): The maximum number of iterations to run the algorithm. Default is 10.

    Returns:
    np.ndarray: A 2D array where each pixel is labeled with an integer representing the superpixel it belongs to.

    Learning Notes (line by line):
    Sets height and width of the 2D image
    Sets spacing between superpixels based on the number of segments
    Initializes a list to store the centers of the superpixels
    Converts the list of centers to a NumPy array
    Sets the number of clusters to the number of centers
    Creates array labels to store the labels of each pixel in the image
    Each iteration, resets distances between pixels and centers to infinity
    Loops through each center by index and defines a search window around the center
    Loops and calculates the distance between each pixel and the center
        Calculates the spatial distance from the edge of the window and the superpixel center
    Calcultes the intensity difference between the pixel and the center, rescaled to 0-100
    Calculates the combined distance metric, to balance color and spatial distance
        Spatial distance is scaled by compactness / S (the grid step), as in the original SLIC paper,
        so compactness controls how square the superpixels are regardless of image size
    Reassign superpixel cnter if D is less than the current distance
        D is a measure of similarity between the intensity difference and the patial distance between the center and the pixel
    Update superpixel centers based on the mean of the pixels assigned to each center to be run through next iteration
    """
    h, w = image.shape
    S = int(np.sqrt(h * w / n_segments)) 
    centers = []
    for y in range(S//2, h, S):
        for x in range(S//2, w, S):
            centers.append([y, x, int(image[y, x])])
    centers = np.array(centers)
    n_clusters = len(centers)
    labels = -np.ones((h, w), dtype=np.int32)
    for _ in range(max_iter):
        distances = np.full((h, w), np.inf)   # reset each pass so pixels can move to a closer, updated center
        for idx, (cy, cx, ci) in enumerate(centers):
            y_start = max(0, cy - S)
            y_end = min(h, cy + S)
            x_start = max(0, cx - S)
            x_end = min(w, cx + S)
            y_start, y_end = int(y_start), int(y_end)
            x_start, x_end = int(x_start), int(x_end)
            for y in range(y_start, y_end):
                for x in range(x_start, x_end):
                    ds = np.sqrt((y - cy)**2 + (x - cx)**2)
                    dc = abs(int(image[y, x]) - ci) * 100 / 255     # intensity on a 0-100 scale, like CIELAB lightness
                    D = np.sqrt(dc**2 + (ds * compactness / S)**2)  # standard SLIC distance
                    if D < distances[y, x]:
                        distances[y, x] = D
                        labels[y, x] = idx
        new_centers = np.zeros_like(centers, dtype=np.float64)
        counts = np.zeros(n_clusters, dtype=np.int32)
        for y in range(h):
            for x in range(w):
                label = labels[y, x]
                if label >= 0:
                    new_centers[label, 0] += y
                    new_centers[label, 1] += x
                    new_centers[label, 2] += image[y, x]
                    counts[label] += 1
        for i in range(n_clusters):
            if counts[i] > 0:
                new_centers[i] /= counts[i]
        centers = new_centers
    return labels

def blob_log(image, min_sigma=2, max_sigma=30, num_sigma=10, threshold=0.02):
    """
    Detects blobs in a grayscale image using the Laplacian of Gaussian (LoG) method.

    Parameters:
    image (np.ndarray): A 2D grayscale image represented as a NumPy array.
    min_sigma (float, optional): Minimum standard deviation for Gaussian kernel. Default is 2.
    max_sigma (float, optional): Maximum standard deviation for Gaussian kernel. Default is 30.
    num_sigma (int, optional): Number of intermediate values of standard deviations to consider. Default is 10.
    threshold (float, optional): Absolute intensity threshold for blob detection. Default is 0.02.

    Returns:
    List[Tuple[int, int, float]]: A list of detected blobs, where each blob is represented by a tuple (y, x, radius).

    Learning Notes (line by line):
    Initializes the height and width of the image
    Rescales intensities to 0-1 so the threshold means the same thing for every image
    Initializes the sigma list with evenly spaced values between min_sigma and max_sigma
        Variance in sigma values allows for detection of blobs of different sizes
    Creates a list to store LoG filtered images
    Loops through each sigma value and applies Gaussian filter, highlighting areas of rapid intensity change
    Multiplies by sigma squared to normalize responses across scales, and takes the absolute value to make all values positive
    Stores all LoG images into the log_images list
    Loops through the log-images list and compares 3 consecutive slices
    Checks if the current slice is a local maximum and above the threshold
    Find each local maxima and computes the radius of the blob
        sigma * sqrt(2) is the common gaussian approximation for the radius of the blob
    Returns the list of blobs
    """
    h, w = image.shape
    image = image.astype(np.float64) / 255.0   # threshold is defined for intensities in 0-1
    sigma_list = np.linspace(min_sigma, max_sigma, num_sigma)
    log_images = []
    for sigma in sigma_list:
        log = ndi.gaussian_laplace(image, sigma=sigma)
        log = (sigma ** 2) * np.abs(log)       # scale-normalized response, positive for bright and dark blobs
        log_images.append(log)
    log_stack = np.stack(log_images, axis=-1)
    coordinates = []
    for i in range(1, num_sigma - 1):
        slice_prev = log_stack[..., i - 1]
        slice_curr = log_stack[..., i]
        slice_next = log_stack[..., i + 1]
        local_max = (
            (slice_curr > threshold) &
            (slice_curr > slice_prev) &
            (slice_curr > slice_next) &
            (slice_curr == ndi.maximum_filter(slice_curr, size=3))
        )
        ys, xs = np.where(local_max)
        for y, x in zip(ys, xs):
            coordinates.append((y, x, sigma_list[i] * np.sqrt(2)))
    return coordinates

def color_segments(image, segments):
    """
    Colors each segment in the image with the mean intensity of its pixels.

    Parameters:
    image (np.ndarray): A 2D grayscale image as a NumPy array.
    segments (np.ndarray): A 2D array of the same shape as `image`, where each pixel's value is the label of the segment it belongs to.

    Returns:
    np.ndarray: A 2D uint8 image where each segment is filled with its mean intensity.

    Learning Notes (line by line):
    Creates an oututarray of zeros the same shape as the image, and sets all pixel values to 0 - 255 grayscale range
    Loop through each unique segment label in the segments array
    Marks each corresponding pixel that belong to label as true
    Extracts average of all pixel values in the segment and assigns the value to the whole segment
    Returns the output array with the mean intensity values filled in for each segment
    """
    output = np.zeros_like(image, dtype=np.uint8)
    for label in np.unique(segments):
        mask = segments == label
        mean_val = np.mean(image[mask])
        output[mask] = int(mean_val)
    return output

def plot_segmentation(image, felz_output, slic_output, blobs):
    """
    Visualizes the results of different image segmentation techniques:
    Felzenszwalb segmentation, SLIC superpixels, and Laplacian of Gaussian blob detection.

    Parameters:
    image (np.ndarray): A 2D grayscale image represented as a NumPy array.
    felz_output (np.ndarray): A 2D array with segment labels produced by the Felzenszwalb algorithm.
    slic_output (np.ndarray): A 2D array with segment labels produced by the SLIC algorithm.
    blobs (List[Tuple[float, float, float]]): A list of tuples representing detected blobs, where each blob is (y, x, radius).

    Returns:
    None

    Learning Notes (line by line):
    Finds all unique segment labels for felzenszwalb output, creates a dictionary mapping of that output, the applies the mapping to the whole segmentation map normalized to 0-255
    Uses the color_segments function to color the SLIC segments
    Creates an empty array to store the blob detection resultsm then apply the blobs with their appropriate radii
        Creates 100 points in the circle radius to the draw the blob and converts r into its cartesian offsets to create circles
    Outputs the original image, felzenszwalb segmentation, SLIC superpixels, and blob detection results in a 4-panel plot
    """
    felz_unique = np.unique(felz_output)
    felz_map = {id_: i for i, id_ in enumerate(felz_unique)}
    felz_colored = np.vectorize(felz_map.get)(felz_output)
    slic_colored = color_segments(image, slic_output)
    blob_colored = np.zeros_like(image, dtype=np.uint8)
    for (y, x, r) in blobs:
        y, x = int(y), int(x)
        if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
            blob_colored[y, x] = 255
            for angle in np.linspace(0, 2 * np.pi, 100):
                dx = int(r * np.cos(angle))
                dy = int(r * np.sin(angle))
                if 0 <= x + dx < image.shape[1] and 0 <= y + dy < image.shape[0]:
                    blob_colored[y + dy, x + dx] = 255

    plt.figure(figsize=(15, 6))
    plt.subplot(1, 4, 1)
    plt.imshow(image, cmap='gray')
    plt.title("Original Image")
    plt.axis('off')

    plt.subplot(1, 4, 2)
    plt.imshow(felz_colored, cmap='tab20')
    plt.title("Felzenszwalb Segmentation")
    plt.axis('off')

    plt.subplot(1, 4, 3)
    plt.imshow(slic_colored, cmap='gray')
    plt.title("SLIC Superpixels")
    plt.axis('off')

    plt.subplot(1, 4, 4)
    plt.imshow(blob_colored)
    plt.title("Blob Detection")
    plt.axis('off')

    plt.tight_layout()
    plt.show()
