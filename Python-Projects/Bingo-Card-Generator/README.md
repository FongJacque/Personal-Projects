# Bingo Card Generator

Generates a random 5 × 5 bingo card, prints it to the terminal and draws it
as an image.

- **Numbers or words:** by default the card uses numbers 1–75. Set
  `pool_type = "strings"` in `bingo_game()` to build a card from your own
  word list (it needs at least 25 entries). Handy for classroom or party
  games.
- **No repeats:** each value is drawn once from a copy of the pool.

## How to run

```bash
pip install pillow
python bingo_card.py
```

The card opens in your image viewer. To also save it, call
`create_bingo_image(card, "bingo_card.png")`.

## Changes since the original version

- Renamed to `bingo_card.py` and added a `__main__` guard, so the functions
  can be imported without starting a game.
- The image is shown but no longer saved automatically.
- Uses a readable font when Arial isn't installed (DejaVu Sans, then Pillow's
  built-in font at full size).
