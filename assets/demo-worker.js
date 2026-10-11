// =====================================================================
// demo-worker.js — runs a Python project in the visitor's browser with
// Pyodide (Python compiled to WebAssembly).
//
// Works for any project: it detects the packages the code imports and
// loads them, shows plots (matplotlib, OpenCV, Pillow) and any image the
// script saves. Everything lives in memory in a background worker:
// nothing is installed on the visitor's computer, and refreshing the
// page wipes it.
// =====================================================================

const PYODIDE_URL = "https://cdn.jsdelivr.net/pyodide/v0.27.7/full/";
importScripts(PYODIDE_URL + "pyodide.js");

let pyodide = null;
const send = (type, data) => postMessage({ type, data });

const DOWNLOAD_HELP = "This is usually a brief network hiccup, a VPN or firewall, or an ad/privacy blocker " +
    "blocking cdn.jsdelivr.net (where the browser downloads Python). Check that, then press Run again.";

const wait = ms => new Promise(resolve => setTimeout(resolve, ms));

async function startPython() {
    send("status", "Loading Python…");
    try {
        pyodide = await loadPyodide({ indexURL: PYODIDE_URL });
    } catch (e) {
        throw new Error("Couldn't download Python. " + DOWNLOAD_HELP);
    }
    pyodide.setStdout({ batched: line => send("stdout", line) });
    pyodide.setStderr({ batched: line => {
        if (/building the font cache/i.test(line)) return;   // routine first-run setup message, not an error
        send("stderr", line);
    } });
    pyodide.setStdin({ stdin: () => null });   // demos can't ask for typed input; input() raises EOFError
    pyodide.registerJsModule("demo_bridge", {
        image: (mime, base64) => send("image", { mime, base64 }),
    });
}

// Runs before the project: redirects every "show a picture" call to the page
const SETUP = `
import sys, os, io, base64
from demo_bridge import image as _send_image

def _send_png(png_bytes):
    _send_image("image/png", base64.b64encode(png_bytes).decode())

# matplotlib: plt.show() sends each open figure as a PNG
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    def _plt_show(*args, **kwargs):
        for num in plt.get_fignums():
            buf = io.BytesIO()
            plt.figure(num).savefig(buf, format="png", dpi=110, bbox_inches="tight")
            _send_png(buf.getvalue())
        plt.close("all")
    plt.show = _plt_show
except ImportError:
    plt = None

# OpenCV: cv2.imshow() sends the image; window functions do nothing
try:
    import cv2
    def _cv_imshow(name, img):
        ok, png = cv2.imencode(".png", img)
        if ok:
            _send_png(png.tobytes())
    cv2.imshow = _cv_imshow
    cv2.waitKey = lambda *a, **k: -1
    cv2.destroyAllWindows = lambda *a, **k: None
    cv2.destroyWindow = lambda *a, **k: None
    cv2.namedWindow = lambda *a, **k: None
except ImportError:
    pass

# Pillow: Image.show() sends the image
try:
    from PIL import Image
    def _pil_show(self, *args, **kwargs):
        buf = io.BytesIO()
        self.save(buf, format="PNG")
        _send_png(buf.getvalue())
    Image.Image.show = _pil_show
except ImportError:
    pass
`;

const RUN = `
import runpy, sys, os
# Same as running "python <entry>" from inside the project folder
os.chdir("/project")
_script_dir = os.path.dirname(os.path.join("/project", ENTRY))
for _p in ("/project", _script_dir):
    if _p not in sys.path:
        sys.path.insert(0, _p)
sys.argv = [ENTRY] + list(ARGS)   # e.g. ["demo.py", "user_input/upload.jpg"]
EXIT_MESSAGE = None
try:
    runpy.run_path("/project/" + ENTRY, run_name="__main__")
except SystemExit as _exit:
    # sys.exit("message") is how a script reports a problem to its user; sys.exit() / sys.exit(0) is success
    if isinstance(_exit.code, str):
        EXIT_MESSAGE = _exit.code
    elif _exit.code not in (None, 0):
        EXIT_MESSAGE = f"The script stopped with exit code {_exit.code}."
finally:
    if plt is not None:
        plt.show()     # any figures the script made but never showed
`;

// After the run: list image files the script created (savefig, imwrite, ...)
const NEW_IMAGES = `
import os
_found = []
for root, _, files in os.walk("/project"):
    for f in files:
        path = os.path.join(root, f)
        if os.path.splitext(f)[1].lower() in IMAGE_EXT and os.path.relpath(path, "/project") not in ORIGINAL:
            _found.append(path)
sorted(_found)
`;

const IMAGE_TYPES = { ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".gif": "image/gif", ".webp": "image/webp", ".svg": "image/svg+xml" };

function toBase64(bytes) {
    let binary = "";
    for (let i = 0; i < bytes.length; i += 0x8000) {
        binary += String.fromCharCode.apply(null, bytes.subarray(i, i + 0x8000));
    }
    return btoa(binary);
}

onmessage = async (event) => {
    // files: { "relative/path.ext": ArrayBuffer }, fetched by the page
    const { entry, files, args, packages } = event.data;
    try {
        if (!pyodide) await startPython();

        // Fresh, empty in-memory project folder for every run
        pyodide.runPython(`
import shutil, sys
shutil.rmtree("/project", ignore_errors=True)
for _name, _mod in list(sys.modules.items()):
    if (getattr(_mod, "__file__", None) or "").startswith("/project/"):
        del sys.modules[_name]
`);
        for (const [path, buffer] of Object.entries(files)) {
            const dir = ("/project/" + path).split("/").slice(0, -1).join("/");
            pyodide.FS.mkdirTree(dir);
            pyodide.FS.writeFile("/project/" + path, new Uint8Array(buffer));
        }

        // Find the code that actually runs: the entry file plus local modules it imports
        pyodide.globals.set("ENTRY", entry);
        pyodide.globals.set("EXTRA", pyodide.toPy(packages || []));
        pyodide.runPython(`
import os
from pyodide.code import find_imports

def _local_file(name):
    for candidate in (name.replace(".", "/") + ".py", name.replace(".", "/") + "/__init__.py"):
        for base in ("", os.path.dirname(ENTRY)):
            path = os.path.normpath(os.path.join(base, candidate))
            if os.path.exists("/project/" + path):
                return path
    return None

SOURCES, _seen, _todo, IMPORTS = [], set(), [ENTRY], set()
while _todo:
    _f = _todo.pop()
    if _f in _seen:
        continue
    _seen.add(_f)
    _code = open("/project/" + _f, encoding="utf-8", errors="replace").read()
    SOURCES.append(_code)
    for _name in find_imports(_code):
        _local = _local_file(_name)
        if _local:
            _todo.append(_local)
        else:
            IMPORTS.add(_name.split(".")[0])
`);

        // Load those packages (Pyodide ships numpy, pandas, scikit-learn, OpenCV, ...).
        // Big downloads sometimes get cut off, so failed ones are retried.
        const sources = pyodide.globals.get("SOURCES").toJs();
        for (let attempt = 1; attempt <= 3; attempt++) {
            send("status", attempt === 1 ? "Loading packages…" : `Download interrupted, retrying packages (try ${attempt} of 3)…`);
            let failed = false;
            for (const code of sources) {
                await pyodide.loadPackagesFromImports(code, {
                    messageCallback: () => {},
                    errorCallback: () => { failed = true; },
                });
            }
            if (!failed) break;
            if (attempt === 3) throw new Error("Couldn't download some of the Python packages this project needs. " + DOWNLOAD_HELP);
            await wait(1500 * attempt);
        }

        // Anything not bundled with Pyodide: try installing it from PyPI
        const missing = pyodide.runPython(`
import importlib.util
sorted(set(EXTRA) | {n for n in IMPORTS if importlib.util.find_spec(n) is None})
`).toJs();
        if (missing.length) {
            send("status", "Installing " + missing.join(", ") + "…");
            await pyodide.loadPackage("micropip", { messageCallback: () => {} });
            const micropip = pyodide.pyimport("micropip");
            for (const name of missing) {
                try { await micropip.install(name); }
                catch (e) { send("stderr", `Couldn't install "${name}" in the browser.`); }
            }
        }

        send("status", `Running ${entry}…`);
        await pyodide.runPythonAsync(SETUP);
        pyodide.globals.set("ARGS", pyodide.toPy(args || []));
        let exitMessage = null;
        try {
            await pyodide.runPythonAsync(RUN);
            exitMessage = pyodide.globals.get("EXIT_MESSAGE");
        } finally {
            pyodide.globals.set("ORIGINAL", pyodide.toPy(Object.keys(files)));
            pyodide.globals.set("IMAGE_EXT", pyodide.toPy(Object.keys(IMAGE_TYPES)));
            for (const path of pyodide.runPython(NEW_IMAGES).toJs()) {
                const ext = path.slice(path.lastIndexOf(".")).toLowerCase();
                send("image", { mime: IMAGE_TYPES[ext], base64: toBase64(pyodide.FS.readFile(path)) });
            }
        }
        if (exitMessage) send("error", exitMessage);   // shown as-is, no traceback
        else send("done");
    } catch (err) {
        send("error", String(err.message || err));
    }
};
