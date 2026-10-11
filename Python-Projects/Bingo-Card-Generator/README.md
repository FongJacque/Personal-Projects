# Bingo Card Generator

Generates a random 5 × 5 bingo card, prints it to the terminal and draws it
as an image.

- **Numbers or words:** by default the card uses numbers 1–75. Give it a
  text file of your own words or short phrases (one per line, or separated
  by commas) to make a custom card, handy for classroom, party or meeting
  games. It needs at least 25 different entries; with more, 25 are picked at
  random.
- **Text that fits:** each square's text is sized and wrapped to fit, breaking
  at spaces or hyphens, so phrases work as well as single words.
- **No repeats:** each value is drawn once from a copy of the pool.

## How to run

```bash
pip install pillow
python bingo_card.py                # numbers 1-75
python bingo_card.py my_words.txt   # your own words
```

On the website, type or paste your words into the box above **Run demo**.

The card opens in your image viewer. To also save it, call
`create_bingo_image(card, "bingo_card.png")`.

## Changes since the original version

- Renamed to `bingo_card.py` and added a `__main__` guard, so the functions
  can be imported without starting a game.
- The image is shown but no longer saved automatically.
- Uses a readable font when Arial isn't installed (DejaVu Sans, then Pillow's
  built-in font at full size).
- Custom word lists from a file, with text fitted and wrapped to each square.
- The B-I-N-G-O letters are centered over their columns.
