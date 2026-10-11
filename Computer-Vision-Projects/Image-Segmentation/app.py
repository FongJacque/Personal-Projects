# Jacque Fong
"""
app.py — desktop app: pick an image and see all three segmentations.

Run:  python app.py   then click "Upload Image & Segment".
"""

from tkinter import filedialog, Tk, Button
from PIL import Image
import numpy as np

from segmentation import felzenszwalb_segmentation, slic_grayscale, blob_log, plot_segmentation


def load_image():
    """
    Opens a file dialog to select an image file, loads it, and applies segmentation algorithms.

    Parameters:
    None

    Returns:
    None

    Learning Notes:
    Tkinter file dialog to select an image file of 3 types: jpg, png, bmp
    If input file is valid, Pillow opens the image and converts it to grayscale
    Converts the image to a NumPy array for processing
    Applies the Felzenszwalb segmentation, SLIC superpixel segmentation, and blob detection algorithms
    Calls the plot_segmentation function to visualize the results
    """
    path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png *.bmp")])
    if path:
        img = Image.open(path).convert('L')
        image_np = np.array(img)
        felz_segmented = felzenszwalb_segmentation(image_np, k=300)
        slic_segmented = slic_grayscale(image_np, n_segments=200)
        blobs = blob_log(image_np, min_sigma=1, max_sigma=30, num_sigma=10, threshold=0.1)
        plot_segmentation(image_np, felz_segmented, slic_segmented, blobs)

def launch_gui():
    """
    Initializes a basic GUI.

    Parameters:
    None

    Returns:
    None

    Learning Notes (line by line):
    Creates Tkinter window
    Sets window title
    Sets window size
    Creates interactive button with the function of intaking an image
    Places button in the center of the window with 30 pizels of space around it
    Starts the Tkinter main loop
    """
    root = Tk()
    root.title("Image Segmentation: Felzenszwalb + SLIC + Blob Detection")
    root.geometry("300x100")
    btn = Button(root, text="Upload Image & Segment", command=load_image)
    btn.pack(pady=30)
    root.mainloop()

if __name__ == "__main__":
    launch_gui()
