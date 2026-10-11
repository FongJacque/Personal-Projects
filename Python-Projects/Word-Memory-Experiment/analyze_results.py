"""
analyze_results.py — summarizes the word recognition experiment.

Reads every results file in results/ (one per participant per round), then:
  1. Classifies each test word: shown & associated with the scene,
     shown & unrelated, or not shown (a lure).
  2. Computes recognition ("hit") rates per participant and per list length.
  3. Compares associated vs unrelated words with a paired t-test.
  4. Plots the results.

Nothing is written to disk unless SAVE_SUMMARY is set to True.

Run:  python analyze_results.py
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import openpyxl  # noqa: F401  (pandas uses it to read .xlsx files)

RESULTS_DIR = Path(__file__).parent / "results"
SAVE_SUMMARY = False      # True writes summary.xlsx next to this file

FILENAME = re.compile(r"(?P<participant>\d+) Test (?P<position>\d) Results \((?P<associated>\d+) Associated Words\)")


def load_results(folder):
    """One row per test word, with participant and round details from the file name."""
    frames = []
    for path in sorted(folder.glob("*.xlsx")):
        match = FILENAME.match(path.stem)
        if not match:
            continue
        df = pd.read_excel(path)
        df["Participant"] = match["participant"]
        df["Round"] = int(match["position"])                  # order the round was taken in
        df["WordsShown"] = 2 * int(match["associated"])       # half associated, half unrelated
        frames.append(df)
    if not frames:
        raise SystemExit(f"No results files found in {folder}")
    data = pd.concat(frames, ignore_index=True)

    data["Type"] = np.select(
        [data["WasShown"] & data["Associated"], data["WasShown"] & ~data["Associated"]],
        ["Shown, associated", "Shown, unrelated"],
        default="Not shown (lure)",
    )
    return data


data = load_results(RESULTS_DIR)
n_people = data["Participant"].nunique()
print(f"{n_people} participants, {data.groupby('Participant')['Round'].nunique().min()} rounds each, "
      f"{len(data):,} test answers\n")

shown = data[data["WasShown"]]

# ---- 1. Per participant: recognition of shown words, plus lure false alarms ----
per_person = (
    data.groupby(["Participant", "Type"])["UserAnsweredYes"].mean()
        .unstack("Type")
        .rename(columns={"Shown, associated": "Hit rate: associated",
                         "Shown, unrelated": "Hit rate: unrelated",
                         "Not shown (lure)": "False alarm: lures"})
)
per_person["Overall accuracy"] = data.groupby("Participant")["Correct"].mean()
print("Per participant (share answered 'yes'; accuracy = share answered correctly)")
print(per_person.round(2).to_string(), "\n")

# ---- 2. By list length: mean hit rate across participants ----
person_length = shown.groupby(["Participant", "WordsShown", "Type"])["UserAnsweredYes"].mean().reset_index()
by_length = person_length.groupby(["WordsShown", "Type"])["UserAnsweredYes"].agg(["mean", "sem"]).unstack("Type")
print("Hit rate by number of words shown (mean across participants)")
print(by_length["mean"].round(2).to_string(), "\n")

# ---- 3. Associated vs unrelated: paired t-test on each person's hit rates ----
assoc = per_person["Hit rate: associated"]
unrel = per_person["Hit rate: unrelated"]
t, p = stats.ttest_rel(assoc, unrel)
diff = (assoc - unrel).mean()
print("Associated vs unrelated words (paired t-test across participants)")
print(f"  mean hit rate: associated {assoc.mean():.2f}, unrelated {unrel.mean():.2f}, difference {diff:+.2f}")
print(f"  t({n_people - 1}) = {t:.2f}, p = {p:.3f}")
print("  " + ("Associated words were recognized more often."
              if p < 0.05 and diff > 0 else
              "No reliable difference with this sample size."))
print("\nNote: lure false-alarm rates are shown for completeness. In this data set the lures were drawn")
print("from the wrong word list (a bug since fixed in experiment.py), so they aren't a valid measure.")

if SAVE_SUMMARY:
    out = Path(__file__).parent / "summary.xlsx"
    with pd.ExcelWriter(out) as writer:
        per_person.to_excel(writer, sheet_name="Per participant")
        by_length["mean"].to_excel(writer, sheet_name="By list length")
    print(f"\nSaved {out.name}")

# ---- 4. Plots ----
colors = {"Shown, associated": "#8b5a2b", "Shown, unrelated": "#9aa7b4"}
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4), gridspec_kw={"width_ratios": [1.3, 1]})

lengths = by_length.index.to_numpy()
width = 0.36
for i, kind in enumerate(colors):
    ax1.bar(np.arange(len(lengths)) + (i - 0.5) * width, by_length["mean"][kind], width,
            yerr=by_length["sem"][kind], capsize=4, color=colors[kind], label=kind.replace("Shown, ", "").capitalize())
ax1.set_xticks(np.arange(len(lengths)), [f"{n} words" for n in lengths])
ax1.set_ylim(0, 1)
ax1.set_ylabel("Hit rate (share of shown words recognized)")
ax1.set_title("Recognition by list length")
ax1.legend(frameon=False)

for person in per_person.index:
    ax2.plot([0, 1], [assoc[person], unrel[person]], color="#c9b8a6", marker="o", zorder=1)
ax2.plot([0, 1], [assoc.mean(), unrel.mean()], color="#2b231d", marker="o", linewidth=3, zorder=2, label="Mean")
ax2.set_xticks([0, 1], ["Associated", "Unrelated"])
ax2.set_xlim(-0.4, 1.4)
ax2.set_ylim(0, 1)
ax2.set_title(f"Each participant (n = {n_people})\npaired t = {t:.2f}, p = {p:.3f}")
ax2.legend(frameon=False, loc="lower left")

for ax in (ax1, ax2):
    ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.show()
