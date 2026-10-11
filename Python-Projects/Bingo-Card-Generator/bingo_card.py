"""
bingo_card.py — generates a random 5x5 bingo card.

Prints the card and draws it as an image, from numbers 1-75 or from your own
list of words or short phrases.

Run:
    python bingo_card.py                 # numbers 1-75
    python bingo_card.py my_words.txt    # your own words: one per line, or separated by commas

You can also switch pool_type in bingo_game() to "strings" to use the
built-in fruit list.
"""

import random
import re
import sys
from functools import lru_cache
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

MAX_ENTRY_LENGTH = 40     # characters per word or phrase; longer ones are cut


def generate_bingo_card(pool):
    if len(pool) < 25:
        raise ValueError("Pool must contain at least 25 elements")
    card = [['' for _ in range(5)] for _ in range(5)]
    available_positions = [(row, col) for row in range(5) for col in range(5)]
    pool_copy = pool[:]
    for pos in available_positions:
        row, col = pos
        choice = random.choice(pool_copy)
        card[row][col] = choice
        pool_copy.remove(choice)
    return card


def read_word_list(path):
    """
    Reads words or phrases from a text file: one per line, or separated by commas.
    Drops blanks and repeats (ignoring capitalization), keeping the first spelling.
    """
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    entries, seen = [], set()
    for line in text.splitlines():
        for part in line.split(","):
            entry = " ".join(part.split())[:MAX_ENTRY_LENGTH]
            if entry and entry.lower() not in seen:
                seen.add(entry.lower())
                entries.append(entry)
    return entries


@lru_cache(maxsize=None)
def load_font(size):
    """Arial if installed, then DejaVu Sans (common on Linux), then Pillow's built-in font."""
    for name in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except IOError:
            pass
    try:
        return ImageFont.load_default(size=size)   # Pillow 10.1+
    except TypeError:
        return ImageFont.load_default()


def _tokens(text):
    """Words, with hyphenated words split after the hyphen so they can wrap there."""
    return re.findall(r"[^\s-]+-+|\S+", text)


def _join(tokens):
    """Joins tokens with spaces, except straight after a hyphen."""
    out = ""
    for tok in tokens:
        out += tok if (not out or out.endswith("-")) else " " + tok
    return out


def _break_long(text, font, draw, max_width):
    """Last resort: wraps at any character, hyphenating inside words."""
    lines, current = [], ""
    for ch in text:
        trial = current + ch
        if draw.textlength(trial + "-", font=font) <= max_width or not current:
            current = trial
        else:
            lines.append(current if current.endswith(" ") else current + "-")
            current = ch.lstrip()
    lines.append(current)
    return [l.strip() for l in lines if l.strip()]


def fit_text(draw, text, max_width, max_height, largest=30, smallest=11):
    """
    Finds the largest font that fits text in a cell, trying one line, then two,
    then three lines broken at spaces or hyphens. Returns (font, lines).
    """
    tokens = _tokens(text)

    def splits(n):
        """Every way to break the tokens into n lines, most balanced first."""
        if n == 1:
            return [[text]]
        if len(tokens) < n:
            return []
        options = []
        if n == 2:
            for i in range(1, len(tokens)):
                options.append([_join(tokens[:i]), _join(tokens[i:])])
        else:
            for i in range(1, len(tokens) - 1):
                for j in range(i + 1, len(tokens)):
                    options.append([_join(tokens[:i]), _join(tokens[i:j]), _join(tokens[j:])])
        return sorted(options, key=lambda lines: max(len(l) for l in lines))

    for size in range(largest, smallest - 1, -1):
        font = load_font(size)
        line_height = size * 1.15
        for n in (1, 2, 3):
            if n * line_height > max_height:
                break
            for lines in splits(n):
                if all(draw.textlength(line, font=font) <= max_width for line in lines):
                    return font, lines
    # A very long word: break it across lines
    for size in range(16, smallest - 1, -1):
        font = load_font(size)
        lines = _break_long(text, font, draw, max_width)
        if len(lines) * size * 1.15 <= max_height:
            return font, lines
    # Still too long: smallest font, cut with an ellipsis
    font = load_font(smallest)
    cut = text
    while cut and draw.textlength(cut + "…", font=font) > max_width:
        cut = cut[:-1]
    return font, [cut.rstrip() + "…"]


def create_bingo_image(card, filename=None):
    """Draws the card and shows it. Pass a filename to also save it as an image file."""
    width, height = 500, 550
    cell_size = width // 5
    padding = 8
    background_color = (0, 0, 0)
    text_color = (255, 255, 255)
    border_color = (255, 255, 255)
    img = Image.new('RGB', (width, height), background_color)
    draw = ImageDraw.Draw(img)

    # Header: each letter centered over its column
    header_font = load_font(30)
    for col, letter in enumerate("BINGO"):
        draw.text((col * cell_size + cell_size / 2, 25), letter, font=header_font, fill=text_color, anchor="mm")

    # One text size for the whole card: the largest that fits at least 20 of the 25 cells.
    # The few longest entries shrink on their own instead of shrinking everything.
    inner = cell_size - 2 * padding
    best = sorted(fit_text(draw, str(v), inner, inner)[0].size for row in card for v in row)
    card_size = best[len(best) // 5]

    for row in range(5):
        for col in range(5):
            x1, y1 = col * cell_size, row * cell_size + 50
            x2, y2 = (col + 1) * cell_size, (row + 1) * cell_size + 50
            draw.rectangle([x1, y1, x2, y2], outline=border_color, width=2)
            text = str(card[row][col])
            font, lines = fit_text(draw, text, inner, inner, largest=card_size)
            line_height = font.size * 1.15
            top = y1 + cell_size / 2 - line_height * (len(lines) - 1) / 2
            for i, line in enumerate(lines):
                draw.text((x1 + cell_size / 2, top + i * line_height), line, font=font, fill=text_color, anchor="mm")
    if filename:
        img.save(filename)
    img.show()


def print_bingo_card(card):
    width = max(2, min(14, max(len(str(v)) for row in card for v in row)))
    print("  ".join(f"{letter:^{width}}" for letter in "BINGO"))
    for i in range(5):
        print("  ".join(f"{str(card[i][j])[:width]:>{width}}" for j in range(5)))


def bingo_game(word_file=None):
    pool_type = "numbers"  # Change this to "strings" to use a list of strings
    # If you want to use numbers, this remains as numbers from 1 to 75
    if pool_type == "numbers":
        pool = list(range(1, 76))
    # If you want to use strings, fill in the pool with your list of strings
    elif pool_type == "strings":
        pool = [
            'Apple', 'Banana', 'Cherry', 'Date', 'Elderberry',
            'Fig', 'Grapes', 'Honeydew', 'Kiwi', 'Lemon',
            'Mango', 'Nectarine', 'Orange', 'Papaya', 'Quince',
            'Raspberry', 'Strawberry', 'Tangerine', 'Uva', 'Watermelon',
            'Pineapple', 'Peach', 'Plum', 'Apricot', 'Blackberry'
        ]  # Add more strings as needed

    # A word file given on the command line replaces the pool
    if word_file:
        pool = read_word_list(word_file)
        if len(pool) < 25:
            sys.exit(f"A bingo card needs at least 25 different words or phrases; you entered {len(pool)}.")
        print(f"Using your {len(pool)} words" + (" (25 picked at random)" if len(pool) > 25 else ""))

    print("Welcome to Bingo!")
    bingo_card = generate_bingo_card(pool)
    print_bingo_card(bingo_card)
    create_bingo_image(bingo_card)
    print("\nGame Over!")


if __name__ == "__main__":
    bingo_game(sys.argv[1] if len(sys.argv) > 1 else None)
