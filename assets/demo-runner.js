// =====================================================================
// demo-runner.js — runs any project's demo in the visitor's browser.
//
//   Python (.py entry) → Pyodide, in a background worker (demo-worker.js)
//   R      (.R entry)  → webR
//
// The project's files are fetched from this website. If projects.json
// doesn't list them, every file in the project's folder is used (the list
// comes from GitHub). Nothing is saved on the visitor's computer.
// =====================================================================

const WEBR_URL = "https://webr.r-wasm.org/v0.4.2/";
const IMAGE_TYPES = { ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".gif": "image/gif", ".webp": "image/webp", ".svg": "image/svg+xml" };
const SKIP_FILES = /(^|\/)(readme\.md|\.gitignore|gitignore|\.ds_store)$/i;

export function demoLanguage(demo) {
    if (demo.language) return demo.language.toLowerCase();
    return /\.r$/i.test(demo.entry || "") ? "r" : "python";
}

// ---------- Which files belong to the project ----------

async function listFiles(repo, project) {
    if (project.demo.files) return project.demo.files;

    const cacheKey = "tree-v1-" + project.id;
    try {
        const saved = JSON.parse(sessionStorage.getItem(cacheKey));
        if (saved && Date.now() - saved.time < 30 * 60000) return saved.files;
    } catch (e) { /* no storage */ }

    const res = await fetch(`https://api.github.com/repos/${repo}/git/trees/main?recursive=1`);
    if (!res.ok) throw new Error("Couldn't get the project's file list from GitHub. Try again in a few minutes.");
    const prefix = project.path.replace(/\/$/, "") + "/";
    const files = (await res.json()).tree
        .filter(item => item.type === "blob" && item.path.startsWith(prefix))
        .map(item => item.path.slice(prefix.length))
        .filter(path => !SKIP_FILES.test(path));

    try { sessionStorage.setItem(cacheKey, JSON.stringify({ time: Date.now(), files })); } catch (e) {}
    return files;
}

async function fetchFiles(project, paths) {
    const files = {};
    await Promise.all(paths.map(async path => {
        const url = "/" + project.path.split("/").concat(path.split("/")).map(encodeURIComponent).join("/");
        const res = await fetch(url, { cache: "no-cache" });
        const type = res.headers.get("content-type") || "";
        // A missing file comes back as the site's HTML page instead of the file
        if (!res.ok || (type.includes("text/html") && !/\.html?$/i.test(path))) {
            throw new Error(`Couldn't load ${path} from the site.`);
        }
        files[path] = await res.arrayBuffer();
    }));
    return files;
}

function cleanTraceback(message) {
    // Drop Pyodide's internal frames; keep the part about the project's own code
    const i = message.indexOf('File "/project/');
    return i > 0 ? "Traceback (most recent call last):\n  " + message.slice(i) : message;
}

function toBase64(bytes) {
    let binary = "";
    for (let i = 0; i < bytes.length; i += 0x8000) binary += String.fromCharCode.apply(null, bytes.subarray(i, i + 0x8000));
    return btoa(binary);
}

// ---------- Python ----------

let pyWorker = null;

function runPython(project, files, args, ui) {
    return new Promise(resolve => {
        if (!pyWorker) pyWorker = new Worker("/assets/demo-worker.js");
        pyWorker.onmessage = ({ data: msg }) => {
            if (msg.type === "status") ui.status(msg.data);
            else if (msg.type === "stdout") ui.stdout(msg.data);
            else if (msg.type === "stderr") ui.stderr(msg.data);
            else if (msg.type === "image") ui.image(`data:${msg.data.mime};base64,${msg.data.base64}`);
            else if (msg.type === "done") resolve(true);
            else if (msg.type === "error") { ui.stderr(cleanTraceback(msg.data)); resolve(false); }
        };
        pyWorker.onerror = e => { ui.stderr(e.message || "The demo crashed."); resolve(false); };
        pyWorker.postMessage({ entry: project.demo.entry, files, args, packages: project.demo.packages || [] });
    });
}

// ---------- R ----------

let webR = null;

async function runR(project, files, ui) {
    if (!webR) {
        ui.status("Loading R…");
        const { WebR } = await import(WEBR_URL + "webr.mjs");
        webR = new WebR({ baseUrl: WEBR_URL });
        await webR.init();
    }

    // Packages: anything used via library(), require() or pkg:: that isn't installed yet
    const decoder = new TextDecoder();
    const code = Object.entries(files).filter(([p]) => /\.r$/i.test(p)).map(([, b]) => decoder.decode(b)).join("\n");
    const wanted = new Set(project.demo.packages || []);
    for (const m of code.matchAll(/(?:library|require|requireNamespace)\(\s*["']?([A-Za-z][\w.]*)["']?/g)) wanted.add(m[1]);
    for (const m of code.matchAll(/([A-Za-z][\w.]*)::/g)) wanted.add(m[1]);
    const installed = await (await webR.evalR("rownames(installed.packages())")).toArray();
    const missing = [...wanted].filter(p => !installed.includes(p));
    if (missing.length) {
        ui.status("Installing " + missing.join(", ") + "…");
        await webR.installPackages(missing, { quiet: true });
    }

    // Fresh, empty project folder
    await webR.evalRVoid('unlink("/project", recursive = TRUE)');
    for (const [path, buffer] of Object.entries(files)) {
        const dir = ("/project/" + path).split("/").slice(0, -1).join("/");
        await webR.evalRVoid(`dir.create("${dir}", recursive = TRUE, showWarnings = FALSE)`);
        await webR.FS.writeFile("/project/" + path, new Uint8Array(buffer));
    }

    ui.status(`Running ${project.demo.entry}… (R output appears when it finishes)`);
    const shelter = await new webR.Shelter();
    let ok = true;
    try {
        const result = await shelter.captureR(
            `setwd("/project"); source(${JSON.stringify(project.demo.entry)}, echo = FALSE, print.eval = TRUE)`,
            { captureStreams: true, captureConditions: false, captureGraphics: { width: 800, height: 500 } }
        );
        result.output.forEach(o => o.type === "stderr" ? ui.stderr(o.data) : ui.stdout(o.data));
        for (const bitmap of result.images) {
            const canvas = document.createElement("canvas");
            canvas.width = bitmap.width; canvas.height = bitmap.height;
            canvas.getContext("2d").drawImage(bitmap, 0, 0);
            ui.image(canvas.toDataURL("image/png"));
        }
    } catch (err) {
        ui.stderr(String(err.message || err));
        ok = false;
    } finally {
        // Images the script saved with png(), ggsave(), ...
        const all = await (await webR.evalR('list.files("/project", recursive = TRUE)')).toArray();
        for (const path of all) {
            const ext = path.slice(path.lastIndexOf(".")).toLowerCase();
            if (IMAGE_TYPES[ext] && !(path in files)) {
                ui.image(`data:${IMAGE_TYPES[ext]};base64,${toBase64(await webR.FS.readFile("/project/" + path))}`);
            }
        }
        shelter.purge();
    }
    return ok;
}

// ---------- Public: start / stop ----------

// extra = { files: { "path": ArrayBuffer }, args: ["..."] } for a visitor's own input (optional)
export async function runDemo(repo, project, ui, extra = {}) {
    try {
        ui.status("Fetching project code…");
        const files = await fetchFiles(project, await listFiles(repo, project));
        if (!(project.demo.entry in files)) throw new Error(`The entry file ${project.demo.entry} isn't in the project folder.`);
        Object.assign(files, extra.files || {});      // visitor's file, kept in memory only
        const args = extra.args || [];
        return demoLanguage(project.demo) === "r" ? await runR(project, files, ui) : await runPython(project, files, args, ui);
    } catch (err) {
        ui.stderr(String(err.message || err));
        return false;
    }
}

export function stopDemo() {
    // Shut the language down immediately; a fresh one starts on the next run
    if (pyWorker) { pyWorker.terminate(); pyWorker = null; }
    if (webR) { webR.close(); webR = null; }
}
