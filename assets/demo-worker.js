// =====================================================================
// demo-worker.js — runs a project's Python code in the visitor's browser
// using Pyodide (Python compiled to WebAssembly).
//
// Runs in a background Web Worker so the page stays responsive while
// the code runs. Everything lives in memory: nothing is installed on
// the visitor's computer, and refreshing the page wipes it.
// =====================================================================

const PYODIDE_URL = "https://cdn.jsdelivr.net/pyodide/v0.27.7/full/";
importScripts(PYODIDE_URL + "pyodide.js");

let pyodide = null;

const send = (type, data) => postMessage({ type, data });

async function startPython(packages) {
    send("status", "Loading Python…");
    pyodide = await loadPyodide({ indexURL: PYODIDE_URL });

    // Stream print() output to the page line by line
    pyodide.setStdout({ batched: line => send("stdout", line) });
    pyodide.setStderr({ batched: line => send("stderr", line) });

    // Lets Python hand finished plots back to the page
    pyodide.registerJsModule("demo_bridge", {
        image: base64png => send("image", base64png),
    });

    send("status", "Loading " + packages.join(", ") + "…");
    await pyodide.loadPackage(packages, { messageCallback: () => {} });   // keep install chatter out of the output
}

// Download the project's own source files from this website
async function fetchSource(basePath, files) {
    const sources = {};
    for (const name of files) {
        const res = await fetch(`/${basePath}/${name}`, { cache: "no-cache" });
        const text = await res.text();
        // A missing file comes back as an HTML page, not Python
        if (!res.ok || text.trimStart().startsWith("<")) {
            throw new Error(`Couldn't load ${name} from the site.`);
        }
        sources[name] = text;
    }
    return sources;
}

const RUNNER = `
import sys, os, io, base64, runpy, shutil
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from demo_bridge import image

def _show(*args, **kwargs):
    # Instead of opening a window, send each figure to the page as a PNG
    for num in plt.get_fignums():
        buf = io.BytesIO()
        plt.figure(num).savefig(buf, format="png", dpi=110, bbox_inches="tight")
        image(base64.b64encode(buf.getvalue()).decode())
    plt.close("all")

plt.show = _show
os.chdir("/project")
if "/project" not in sys.path:
    sys.path.insert(0, "/project")
try:
    runpy.run_path("/project/" + ENTRY, run_name="__main__")
finally:
    _show()   # catch any figures the script made but never showed
`;

onmessage = async (event) => {
    const { basePath, demo } = event.data;
    try {
        if (!pyodide) await startPython(demo.packages);

        send("status", "Fetching project code…");
        const sources = await fetchSource(basePath, demo.files);

        // Fresh, empty in-memory folder for every run
        pyodide.runPython(`
import shutil, sys
shutil.rmtree("/project", ignore_errors=True)
for name in list(sys.modules):
    if name in ${JSON.stringify(demo.files.map(f => f.replace(/\.py$/, "")))}:
        del sys.modules[name]
`);
        pyodide.FS.mkdirTree("/project");
        for (const [name, text] of Object.entries(sources)) {
            pyodide.FS.writeFile("/project/" + name, text);
        }

        send("status", `Running ${demo.entry}…`);
        pyodide.globals.set("ENTRY", demo.entry);
        await pyodide.runPythonAsync(RUNNER);
        send("done");
    } catch (err) {
        send("error", String(err.message || err));
    }
};
