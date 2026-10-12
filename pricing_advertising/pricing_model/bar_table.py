from __future__ import annotations

import csv
import os
import re
from pathlib import Path
from typing import TYPE_CHECKING

from pricing_advertising.enums.enums import RoomCategory

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


# ===============================
#   CSV LOADER (wide format)
# ===============================


def _coerce_int(x: str) -> int:
    try:
        return int(float(x))
    except Exception as e:
        raise ValueError(f"Not numeric level: {x!r}") from e


def _coerce_float(x: str) -> float:
    try:
        return float(x)
    except Exception as e:
        raise ValueError(f"Not numeric price: {x!r}") from e


def load_bar_table_wide_csv(csv_path: str | Path) -> dict[RoomCategory, list[float]]:
    p = Path(csv_path)
    if not p.exists():
        raise FileNotFoundError(f"File CSV not found: {p}")

    with p.open("r", encoding="utf-8", newline="") as fh:
        sample = fh.read(2048)
        fh.seek(0)
        # Automatically detects the separator (;, , or \t)
        try:
            dialect = csv.Sniffer().sniff(sample)
        except Exception:
            dialect = csv.excel
        reader = csv.DictReader(fh, dialect=dialect)

        if reader.fieldnames is None:
            raise ValueError("CSV without header.")

        headers = [h.strip() for h in reader.fieldnames]
        header_up = {h: h.strip().upper() for h in headers}

        # Level column
        level_col = None
        for h, up in header_up.items():
            if up in {"LEVEL"}:
                level_col = h
                break
        if level_col is None:
            raise ValueError("A “LEVEL” column is required in the CSV.")

        # Column CATEGORY1..CATEGORY3
        cat_cols: dict[str, str] = {}
        pat = re.compile(r"^CATEGORY\s*([1-3])$", re.IGNORECASE)
        for h, up in header_up.items():
            up_norm = up.replace("_", "").replace(" ", "")
            m = pat.match(up_norm)
            if m:
                cname = f"CATEGORY{m.group(1)}"
                cat_cols[cname] = h

        missing = [f"CATEGORY{i}" for i in (1, 2, 3) if f"CATEGORY{i}" not in cat_cols]
        if missing:
            raise ValueError(f"Columns missing: {', '.join(missing)}")

        rows = []
        for r in reader:
            rows.append({k: v.strip() if isinstance(v, str) else v for k, v in r.items()})

    # Mapping level -> price
    by_cat: dict[str, dict[int, float]] = {f"CATEGORY{i}": {} for i in (1, 2, 3)}
    for r in rows:
        lv = _coerce_int(r[level_col])
        for cname, hdr in cat_cols.items():
            by_cat[cname][lv] = _coerce_float(r[hdr])

    # Sort by level and fill in any gaps
    lists: dict[str, list[float]] = {}
    max_level = max(max(d.keys()) for d in by_cat.values())
    for cname, d in by_cat.items():
        prices: list[float] = []
        last = 0.0
        for lv in range(max_level + 1):
            if lv in d:
                last = d[lv]
            prices.append(last)
        lists[cname] = prices

    # Check that all categories are the same length.
    lengths = {len(v) for v in lists.values()}
    if len(lengths) != 1:
        raise ValueError(f"All categories must have the same number of levels: {lengths}")

    # Maps a roomcategory
    result: dict[RoomCategory, list[float]] = {}
    for i in (1, 2, 3):
        cname = f"CATEGORY{i}"
        result[RoomCategory[cname]] = lists[cname]

    return result


# ===============================
#   SET BAR TABLE FROM CSV
# ===============================

BAR_TABLE: dict[RoomCategory, list[float]] = {}


def set_bar_table_from_csv(csv_path: str | Path | None = None) -> None:
    global BAR_TABLE
    path = csv_path or os.getenv("BAR_TABLE_CSV")
    if not path:
        raise RuntimeError("Specify the CSV path or set BAR_TABLE_CSV.")
    BAR_TABLE = load_bar_table_wide_csv(path)


# ===============================
#   Helper functions
# ===============================


def room_types() -> list[str]:
    return [room.name for room in BAR_TABLE.keys()]


def n_levels() -> int:
    return len(next(iter(BAR_TABLE.values()))) if BAR_TABLE else 0


def max_price() -> float:
    return max((max(prices) for prices in BAR_TABLE.values()), default=0.0)


def get_price(room_type: str, level: int) -> float:
    prices = BAR_TABLE[RoomCategory[room_type]]
    bar_level = max(0, min(level, len(prices) - 1))
    return prices[bar_level]


def as_mapping() -> Mapping[str, Sequence[float]]:
    return {room.name: prices for room, prices in BAR_TABLE.items()}


def bar_table_as_str(max_levels: int | None = None) -> str:
    if not BAR_TABLE:
        return "<BAR_TABLE is empty. Call set_bar_table_from_csv(...) first.>"

    # Ordino le categorie in base al valore dell'enum (CATEGORY1, CATEGORY2, ...)
    room_order = sorted(BAR_TABLE.keys(), key=lambda r: r.value)

    # Numero livelli
    total_levels = len(next(iter(BAR_TABLE.values())))
    n = min(total_levels, max_levels) if max_levels is not None else total_levels

    # Imposto la larghezza delle colonne
    level_col_width = 5
    price_col_width = 10

    # Header
    header = "LEVEL".ljust(level_col_width)
    for room in room_order:
        header += f"{room.name:>{price_col_width}}"
    lines = [header]

    # Righe per ogni livello
    for lvl in range(n):
        row = f"{lvl:<{level_col_width}d}"
        for room in room_order:
            price = BAR_TABLE[room][lvl]
            row += f"{price:>{price_col_width}.2f}"
        lines.append(row)

    return "\n".join(lines)


def print_bar_table(max_levels: int | None = None) -> None:
    print(bar_table_as_str(max_levels))
