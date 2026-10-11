# Word Recognition Memory Experiment

A psychology experiment I designed and ran, plus the analysis of its results.
It asks whether people remember words better when the words fit the scene
they were shown with.

## The experiment

Each round shows a grayscale scene with a list of words underneath. Half
the words are **associated** with the scene (for a snowy cabin: *frost*,
*timber*, *ridge*) and half are **unrelated** (*lemon*, *radio*, *paint*).
After 30 seconds the words disappear and the participant sees test words one
at a time, pressing **Right** if they saw the word and **Left** if they
didn't.

- **Practice round** with a theater scene, then three test rounds in random order.
- **List length varies by round:** 20, 30 or 40 words shown.
- **The test mixes** words that were shown with associated words that
  weren't (lures), to measure false memories of words that fit the scene.

## Results

7 participants, aged 18–20, 3 rounds each, 420 test answers.

| | Associated words | Unrelated words |
|---|---|---|
| Mean hit rate (share of shown words recognized) | 0.73 | 0.63 |

Associated words were recognized about 10 percentage points more often,
consistently across list lengths, but with seven participants the difference
isn't statistically reliable (paired t-test, t(6) = 1.48, p = 0.19). A
larger sample would be needed to confirm it.

## How to run

```bash
pip install pandas matplotlib scipy openpyxl pillow
```

| Command | What happens |
|---|---|
| `python analyze_results.py` | Prints per-participant and per-list-length tables and the t-test, then shows the plots. Uses the data in `results/`; writes nothing unless `SAVE_SUMMARY = True`. |
| `python experiment.py` | Runs the experiment full-screen and saves each round to `results/`. Needs the scene images below. |

### Scene images

`images/practice.jpg` (the theater) is included. The three test scenes
aren't, because I don't have the rights to publish the originals. To run the
experiment, add your own grayscale photos to `images/`:

| File | Scene | Matching word list |
|---|---|---|
| `test1.jpg` | Rural house or farm | roof, yard, farm, gate… |
| `test2.jpg` | Snowy cabin in the mountains | snow, pine, ridge, frost… |
| `test3.jpg` | Lake or river with trees | shore, willow, ripple, heron… |

## Data

`results/` holds one Excel file per participant per round, named
`<participant> Test <round> Results (<n> Associated Words).xlsx`.
Participants are identified only by number. Each row is one test word:

| Column | Meaning |
|---|---|
| `Word` | The test word |
| `WasShown` | Whether it appeared in the list |
| `Associated` | Whether it was a shown word associated with the scene |
| `UserAnsweredYes` | The participant said they saw it |
| `Correct` | The answer matched `WasShown` |

## Known issue in this data, now fixed

In the original code, lures were picked using the round's position (1st,
2nd, 3rd) instead of its scene. So the first round had no lures, and later
rounds drew lures from a different scene's word list. Hit rates for shown
words are unaffected, but the lure false-alarm rates in this data set aren't
a valid measure of false memory. `experiment.py` now draws lures from the
matching scene's list.

## Changes since the original version

- Renamed the scripts to `experiment.py` and `analyze_results.py`.
- Image and results paths are relative to the project folder instead of
  fixed paths on my computer.
- Fixed the lure bug described above.
- `analyze_results.py` replaces the original results reader. It reads
  `results/` directly instead of using a file picker, separates lures from
  unrelated words (the reader counted them together), adds per-list-length
  results, a paired t-test and plots.

## Photo credit

Practice image by [Elijah Mears](https://unsplash.com/photos/CmKxr073k9k) on
Unsplash, converted to grayscale and resized.
