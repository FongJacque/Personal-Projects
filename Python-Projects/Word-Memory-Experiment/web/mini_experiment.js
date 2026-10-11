// =====================================================================
// mini_experiment.js — a one-round web version of experiment.py.
//
// Same procedure as the experiment's practice round: a scene with 20 words
// (10 associated with the scene, 10 unrelated) for 30 seconds, then 20
// yes/no recognition questions mixing shown words with unseen associated
// words (lures). Word selection follows experiment.py, including the lure fix.
//
// The visitor's answers live only in this page's memory; nothing is saved or
// sent anywhere. Their results are compared with the original study's
// participants, whose numbers come from analyze_results.py.
// =====================================================================

// Practice-round word pools, copied from experiment.py
const ASSOCIATED = ["show", "film", "movie", "stage", "act", "play", "scene", "seat", "ticket", "screen",
                    "lights", "sound", "watch", "drama", "story", "star", "enter", "exit", "crowd", "marquee"];
const UNRELATED  = ["shirt", "table", "phone", "train", "bread", "heart", "anger", "purple", "seven", "sock",
                    "climb", "sing", "draw", "ball", "street", "child", "friend", "apple", "funny", "brave"];

// Original study (7 participants, 3 rounds each), from analyze_results.py
const STUDY = {
    participants: [
        { associated: 0.708, unrelated: 0.621, accuracy: 0.683 },
        { associated: 0.640, unrelated: 0.667, accuracy: 0.650 },
        { associated: 0.760, unrelated: 0.533, accuracy: 0.650 },
        { associated: 0.792, unrelated: 0.967, accuracy: 0.850 },
        { associated: 0.704, unrelated: 0.433, accuracy: 0.550 },
        { associated: 0.760, unrelated: 0.467, accuracy: 0.600 },
        { associated: 0.739, unrelated: 0.733, accuracy: 0.733 },
    ],
    mean: { associated: 0.729, unrelated: 0.632, accuracy: 0.674 },
};

const STUDY_SECONDS = 30;
const WORDS_SHOWN = 20;          // half associated, half unrelated
const TEST_ASSOCIATED = 10;      // associated test words: shown ones plus lures
const MIN_SHOWN_ASSOCIATED = 5;  // at least this many of them were actually shown

// ---------- helpers ----------
function shuffle(list) {
    const a = [...list];
    for (let i = a.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [a[i], a[j]] = [a[j], a[i]];
    }
    return a;
}
const sample = (list, n) => shuffle(list).slice(0, n);
const randint = (lo, hi) => lo + Math.floor(Math.random() * (hi - lo + 1));
const pct = x => `${Math.round(x * 100)}%`;

// Builds one round the way experiment.py does
function buildRound() {
    const shownAssociated = sample(ASSOCIATED, WORDS_SHOWN / 2);
    const shownUnrelated  = sample(UNRELATED, WORDS_SHOWN / 2);
    const shown = shuffle([...shownAssociated, ...shownUnrelated]);

    const unseenAssociated = ASSOCIATED.filter(w => !shownAssociated.includes(w));
    let fromShown = randint(MIN_SHOWN_ASSOCIATED, Math.min(shownAssociated.length, TEST_ASSOCIATED));
    const fromLures = Math.min(TEST_ASSOCIATED - fromShown, unseenAssociated.length);
    fromShown = TEST_ASSOCIATED - fromLures;

    const test = shuffle([
        ...sample(shownAssociated, fromShown).map(word => ({ word, type: "associated" })),
        ...sample(unseenAssociated, fromLures).map(word => ({ word, type: "lure" })),
        ...sample(shownUnrelated, 10).map(word => ({ word, type: "unrelated" })),
    ]);
    return { shown, test };
}

// ---------- the widget ----------
export function mount(root, { imageUrl }) {
    let timer = null;
    let keyHandler = null;

    function cleanup() {
        clearInterval(timer);
        if (keyHandler) document.removeEventListener("keydown", keyHandler);
        keyHandler = null;
    }

    function intro() {
        cleanup();
        root.innerHTML = `
            <p>Take one round of the experiment yourself. You'll see a photo with 20 words underneath for
                ${STUDY_SECONDS} seconds. Then you'll see words one at a time: answer <strong>Yes</strong> if it was
                in the list, <strong>No</strong> if it wasn't. It takes about 90 seconds.</p>
            <p class="mx-note">Your answers aren't saved or sent anywhere; they disappear when you leave the page.</p>
            <button class="btn mx-start" type="button"><i class="fa-solid fa-play" aria-hidden="true"></i> Start</button>`;
        root.querySelector(".mx-start").addEventListener("click", study);
    }

    function study() {
        cleanup();
        const round = buildRound();
        root.innerHTML = `
            <div class="mx-study">
                <img class="mx-scene" src="${imageUrl}" alt="Black-and-white photo of a theater with a marquee">
                <ul class="mx-words">${round.shown.map(w => `<li>${w}</li>`).join("")}</ul>
                <div class="mx-timer">
                    <div class="mx-bar"><span style="width:100%"></span></div>
                    <span class="mx-count" aria-live="off">${STUDY_SECONDS} s</span>
                </div>
            </div>`;
        root.scrollIntoView({ behavior: "smooth", block: "start" });
        const bar = root.querySelector(".mx-bar span");
        const count = root.querySelector(".mx-count");
        const start = Date.now();
        timer = setInterval(() => {
            const left = Math.max(0, STUDY_SECONDS - (Date.now() - start) / 1000);
            bar.style.width = `${(left / STUDY_SECONDS) * 100}%`;
            count.textContent = `${Math.ceil(left)} s`;
            if (left <= 0) testPhase(round);
        }, 200);
    }

    function testPhase(round) {
        cleanup();
        const answers = [];
        root.innerHTML = `
            <div class="mx-test">
                <p class="mx-progress"></p>
                <p class="mx-question">Did you see this word?</p>
                <p class="mx-word" aria-live="assertive"></p>
                <div class="mx-answers">
                    <button class="btn btn-outline mx-no" type="button"><i class="fa-solid fa-arrow-left" aria-hidden="true"></i> No</button>
                    <button class="btn mx-yes" type="button">Yes <i class="fa-solid fa-arrow-right" aria-hidden="true"></i></button>
                </div>
                <p class="mx-note">Keyboard: ← No, → Yes</p>
            </div>`;
        const progress = root.querySelector(".mx-progress");
        const wordEl = root.querySelector(".mx-word");

        function show() {
            progress.textContent = `Word ${answers.length + 1} of ${round.test.length}`;
            wordEl.textContent = round.test[answers.length].word;
        }
        function answer(saidYes) {
            answers.push({ ...round.test[answers.length], saidYes });
            if (answers.length === round.test.length) results(answers);
            else show();
        }
        root.querySelector(".mx-no").addEventListener("click", () => answer(false));
        root.querySelector(".mx-yes").addEventListener("click", () => answer(true));
        keyHandler = e => {
            if (e.repeat || e.altKey || e.ctrlKey || e.metaKey) return;
            if (e.key === "ArrowLeft" || e.key.toLowerCase() === "n") { e.preventDefault(); answer(false); }
            if (e.key === "ArrowRight" || e.key.toLowerCase() === "y") { e.preventDefault(); answer(true); }
        };
        document.addEventListener("keydown", keyHandler);
        show();
        root.querySelector(".mx-yes").focus({ preventScroll: true });
    }

    function results(answers) {
        cleanup();
        const of = type => answers.filter(a => a.type === type);
        const yes = list => list.filter(a => a.saidYes).length;
        const assoc = of("associated"), unrel = of("unrelated"), lures = of("lure");
        const you = {
            associated: yes(assoc) / assoc.length,
            unrelated: yes(unrel) / unrel.length,
            accuracy: answers.filter(a => a.saidYes === (a.type !== "lure")).length / answers.length,
        };
        const falseMemories = lures.filter(a => a.saidYes).map(a => a.word);
        const betterInStudy = STUDY.participants.filter(p => p.associated > p.unrelated).length;

        let verdict;
        if (you.associated > you.unrelated) {
            verdict = `You recognized the theater words better than the unrelated ones, like ${betterInStudy} of the
                study's ${STUDY.participants.length} participants.`;
        } else if (you.associated < you.unrelated) {
            verdict = `You recognized the unrelated words better than the theater words. ${STUDY.participants.length - betterInStudy}
                of the study's ${STUDY.participants.length} participants did too, so the effect isn't universal.`;
        } else {
            verdict = `You recognized both kinds of words equally well. In the study, ${betterInStudy} of
                ${STUDY.participants.length} participants did better on the scene-related words.`;
        }

        root.innerHTML = `
            <div class="mx-results">
                <p class="mx-verdict">${verdict}</p>
                <table class="mx-table">
                    <thead><tr><th scope="col"></th><th scope="col">You</th><th scope="col">Study average</th></tr></thead>
                    <tbody>
                        <tr><th scope="row">Theater words you recognized</th><td>${yes(assoc)} of ${assoc.length} (${pct(you.associated)})</td><td>${pct(STUDY.mean.associated)}</td></tr>
                        <tr><th scope="row">Unrelated words you recognized</th><td>${yes(unrel)} of ${unrel.length} (${pct(you.unrelated)})</td><td>${pct(STUDY.mean.unrelated)}</td></tr>
                        <tr><th scope="row">Overall accuracy</th><td>${pct(you.accuracy)}</td><td>${pct(STUDY.mean.accuracy)}</td></tr>
                    </tbody>
                </table>
                ${lures.length ? `<p class="mx-lures"><strong>False memories:</strong> ${lures.length} theater
                    word${lures.length === 1 ? " in the test was" : "s in the test were"} never shown. ${falseMemories.length
                        ? `You thought you saw ${falseMemories.length === lures.length && lures.length === 1 ? "it" : falseMemories.length}: <em>${falseMemories.join(", ")}</em>.`
                        : (lures.length === 1 ? "You correctly said no." : "You correctly said no to all of them.")}
                    Words that fit a scene are easy to "remember" even when they weren't there.</p>` : ""}
                ${dotChart(you)}
                <p class="mx-note">Each grey dot is one study participant; the brown dot is you. The participants did three
                    rounds with different scenes and list lengths, while this is a single 20-word round with the practice
                    scene, so treat the comparison as a rough guide. Nothing you answered was saved.</p>
                <button class="btn btn-outline mx-again" type="button"><i class="fa-solid fa-rotate-right" aria-hidden="true"></i> Try again with new words</button>
            </div>`;
        root.querySelector(".mx-again").addEventListener("click", study);
    }

    // Strip chart: one row per measure, study participants as grey dots, the visitor in brown.
    // Drawn at the container's real width so text stays readable on phones.
    function dotChart(you) {
        const rows = [
            ["associated", "Theater words recognized"],
            ["unrelated", "Unrelated words recognized"],
            ["accuracy", "Overall accuracy"],
        ];
        const W = Math.max(280, Math.round(root.clientWidth || 560));
        const left = 14, right = 18, rowH = 62, top = 6;
        const x = v => left + v * (W - left - right);
        const H = top + rows.length * rowH + 22;
        let svg = `<svg class="mx-chart" width="${W}" height="${H}" viewBox="0 0 ${W} ${H}" role="img"
            aria-label="Your results compared with each study participant">`;
        for (const t of [0, 0.25, 0.5, 0.75, 1]) {
            svg += `<line x1="${x(t)}" x2="${x(t)}" y1="${top + 22}" y2="${H - 20}" class="mx-grid"/>
                    <text x="${x(t)}" y="${H - 4}" class="mx-tick" text-anchor="${t === 0 ? "start" : t === 1 ? "end" : "middle"}">${pct(t)}</text>`;
        }
        rows.forEach(([key, label], i) => {
            const y = top + i * rowH + 40;               // dots
            svg += `<text x="${left}" y="${y - 20}" class="mx-label">${label}</text>`;
            STUDY.participants.forEach((p, j) => {
                svg += `<circle cx="${x(p[key])}" cy="${y + ((j % 3) - 1) * 5}" r="5.5" class="mx-dot"/>`;
            });
            svg += `<line x1="${x(STUDY.mean[key])}" x2="${x(STUDY.mean[key])}" y1="${y - 12}" y2="${y + 12}" class="mx-mean"/>`;
            svg += `<circle cx="${x(you[key])}" cy="${y}" r="8.5" class="mx-you"/>`;
        });
        svg += `</svg>
            <p class="mx-legend"><span><span class="mx-key mx-key-you"></span> You</span>
                <span><span class="mx-key mx-key-dot"></span> Study participant</span>
                <span><span class="mx-key mx-key-mean"></span> Study average</span></p>`;
        return svg;
    }

    intro();
}
