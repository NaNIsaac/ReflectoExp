import re
from datetime import datetime
from pathlib import Path

import pandas as pd
from numpy import ndarray


def append_timestamp(path: Path | str) -> Path:
    """
    Append a timestamp to a path (file or directory) to make it unique.
    The exact naming format is not fixed.
    """
    path = Path(path)
    new_name = f"{path.stem}_{datetime.now():%Y%m%d_%H%M%S}"

    if path.suffix:
        return path.parent / (new_name + path.suffix)
    return path.parent / new_name


def next_unique_file(path: Path | str) -> Path:
    """
    Generate a unique file path.
    - If no file exists with the same name, return the original path.
    - If files exist, append (n) where n = max existing number + 1.

    Example:
        file.txt       -> file.txt        (if doesn't exist)
        file.txt       -> file(1).txt     (if file.txt exists)
        file(1).txt    -> file(2).txt     (if file(1).txt exists)
    """
    path = Path(path)
    parent = path.parent
    stem = path.stem
    suffix = path.suffix

    # Regular expression: Find '(number)' end of the stem.
    pattern = re.compile(rf"^{re.escape(stem)}(?:\((\d+)\))?$")

    # Find same patterns
    existing = [f for f in parent.glob(f"{stem}*{suffix}") if pattern.match(f.stem)]

    if not existing:
        return path

    max_n = 0
    for f in existing:
        m = pattern.match(f.stem)
        if m and m.group(1):
            n = int(m.group(1))
            if n > max_n:
                max_n = n

    return parent / f"{stem}({max_n + 1}){suffix}"


def load_xrr_dat(file: Path | str) -> tuple[ndarray, ndarray]:
    """
    Load 2 column .dat file (2-theta, Intensity).

    Features:
    - Skips lines starting with '#' automatically.
    - Handles scientific notation (e.g., 1.0e-05) automatically.
    - Handles variable whitespace separators.
    - Converts non-numeric garbage to 0.0 safely.
    """

    # comment='#' ignores lines starting with # and trailing # comments after the data
    # sep=r"\s+" accepts tabs and runs of spaces as separators
    df = pd.read_csv(file, header=None, sep=r"\s+", comment="#", names=["tth", "R"])

    # guard against non-numeric garbage mixed into the data
    # (comment='#' removes most of it, but stray text can remain)
    df = df.apply(pd.to_numeric, errors="coerce")

    # NaN (failed conversions) becomes 0.0, then convert to numpy
    # fillna keeps the original row order
    df = df.fillna(0.0)

    return df["tth"].to_numpy(), df["R"].to_numpy()
