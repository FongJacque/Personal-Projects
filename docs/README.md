# Adding a project to the website

Every project on the Projects page comes from one file: `assets/projects.json`.
Adding a project means putting its folder in the repo and adding one entry there.
The project page (activity + runnable demo) is generated automatically.

## 1. Put the project in its section folder

| Section          | Folder                      |
|------------------|-----------------------------|
| R                | `R-Projects/`               |
| Python           | `Python-Projects/`          |
| Computer Vision  | `Computer-Vision-Projects/` |

Example: `Computer-Vision-Projects/Face-Detection/`

## 2. Add an entry to `assets/projects.json`

Add it inside the `"projects": [ ... ]` list, with a comma between entries.

```json
{
  "id": "face-detection",
  "name": "Face Detection",
  "section": "cv",
  "path": "Computer-Vision-Projects/Face-Detection",
  "description": "One or two sentences shown on the project card.",
  "tags": ["Python", "OpenCV"],
  "demo": {
    "entry": "main.py"
  }
}
```

| Field | Required | What it does |
|---|---|---|
| `id` | yes | Short unique name, lowercase with hyphens. Becomes the page address: `project.html?id=face-detection` |
| `name` | yes | Title on the card and project page |
| `section` | yes | `"r"`, `"python"`, or `"cv"` |
| `path` | yes | The project's folder in the repo, exactly as spelled (capitals matter) |
| `description` | yes | Text on the card |
| `tags` | no | Little labels under the description |
| `previousPaths` | no | Old folder names if you ever move the project, so its history isn't lost |
| `demo` | no | Leave it out and the project page shows activity only |

### The `demo` block

| Field | Required | What it does |
|---|---|---|
| `entry` | yes | The file to run, relative to the project folder. `.py` runs as Python, `.R` runs as R |
| `runtime` | no | Shown to visitors, e.g. `"about 30 seconds"` |
| `files` | no | Only needed to limit which files load. By default every file in the folder is loaded |
| `packages` | no | Extra packages the auto-detection misses (rare) |
| `language` | no | `"python"` or `"r"`, only if the entry's extension doesn't say |
| `upload` | no | Lets visitors run the demo on their own file (Python demos). See below |

### Letting visitors use their own file

Add an `upload` block to the demo and the project page shows a button for
picking a file. The file never leaves the visitor's device: it's placed in
the in-browser Python's memory as `user_input/<their file name>` and passed
to the script as its first argument, exactly like running
`python main.py their_file.jpg`. When no file is chosen, the script runs
with no argument, so read it like this and fall back to your sample:

```python
import sys
from pathlib import Path
path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "images" / "sample.jpg"
```

```json
"demo": {
  "entry": "demo.py",
  "upload": {
    "label": "Use your own photo",
    "accept": ".jpg,.jpeg,.png",
    "maxMB": 20,
    "hint": "One sentence on what kinds of files work."
  }
}
```

To report a problem with the visitor's file, call `sys.exit("A short, plain message")`:
the page shows that message on its own, without a traceback.

Packages are detected automatically:
- **Python:** from the `import` lines of the entry file and the local files it imports. NumPy, pandas, matplotlib,
  SciPy, scikit-learn, statsmodels, OpenCV, scikit-image, Pillow, XGBoost and LightGBM are built in.
  Pure-Python packages from PyPI (e.g. seaborn) are installed on the fly.
- **R:** from `library()`, `require()` and `pkg::` in any `.R` file, installed from the webR package repository
  (most CRAN packages, including the tidyverse).

## 3. Make the code demo-friendly

The demo runs the real code in the visitor's browser. It behaves like running
`python main.py` (or `Rscript main.R`) from inside the project folder, with these rules:

- **Use relative paths**, like `data/sample.csv` or `Path(__file__).parent / "data"`.
  Absolute paths from your own computer (`C:\Users\...`, `/home/...`) won't exist.
- **Include the data in the folder**, or generate it in code. CSVs, images and other files all work.
  Keep data small (a few MB) since visitors download it.
- **No typed input.** `input()` / `readline()` can't be answered, so give values in the code.
- **No internet access** from the code (no downloading datasets or calling APIs).
- **Plots and images show automatically:** `plt.show()`, `cv2.imshow()`, `Image.show()`, R plots,
  and any image file the code saves (`savefig`, `cv2.imwrite`, `png()`, `ggsave`).
  Text output from `print()` / `cat()` appears in the console box.
- **Not supported:** PyTorch, TensorFlow, GPUs, webcams, GUI windows (tkinter), or anything needing
  compiled software that isn't built for the browser.
- **Keep it reasonably fast.** Under a minute is ideal; visitors can always press Stop.

## 4. Commit and push

Commit the project folder and the updated `projects.json` together, push, wait for
Cloudflare to deploy, then hard refresh (Ctrl+Shift+R). The project appears in its
section, sorted by its latest commit.

## Testing a demo before you push

From the repo folder, run `python -m http.server` and open
`http://localhost:8000/project.html?id=<your-id>`.

The demo runs your local files, but the list of which files exist comes from GitHub. For a
project you haven't pushed yet, temporarily add a `files` list to its demo block
(e.g. `"files": ["main.py", "data/sample.csv"]`) and remove it before you push. The activity
section also only shows commits after they're pushed.
