"""Strict CSV reader shared by the API, graph, and CLI."""

import csv
import io
from pathlib import Path

import pandas as pd

MAX_BYTES = 512 * 1024 * 1024
MAX_ROWS = 1_000_000
MAX_COLUMNS = 500
# Prevent narrow files from using the entire row limit at the maximum width.
MAX_CELLS = 10_000_000


def read_csv(source: bytes | str | Path) -> pd.DataFrame:
    size = len(source) if isinstance(source, bytes) else Path(source).stat().st_size
    if size > MAX_BYTES:
        raise ValueError("CSV files must be 512 MiB or smaller.")
    binary = io.BytesIO(source) if isinstance(source, bytes) else open(source, "rb")
    try:
        prefix = binary.read(4)
        binary.seek(0)
        encoding = "utf-16" if prefix.startswith((b"\xff\xfe", b"\xfe\xff")) else "utf-8-sig"
        with io.TextIOWrapper(binary, encoding=encoding, newline="") as text:
            sample = text.read(65536)
            if "\x00" in sample:
                raise ValueError("This file contains binary data, not a readable CSV.")
            text.seek(0)
            try:
                dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
            except csv.Error:
                dialect = csv.excel
            reader = csv.reader(text, dialect=dialect, strict=True)
            try:
                headers = next((row for row in reader if row), [])
                normalized = [name.strip().casefold() for name in headers]
                if not headers or any(not name for name in normalized):
                    raise ValueError("CSV needs a non-empty name for every column.")
                if len(set(normalized)) != len(headers):
                    raise ValueError("CSV has duplicate column names. Give each column a unique name.")
                if len(headers) > MAX_COLUMNS:
                    raise ValueError(f"CSV supports at most {MAX_COLUMNS} columns.")
                columns = [[] for _ in headers]
                count = 0
                for row in reader:
                    if not row:
                        continue
                    if len(row) != len(headers):
                        raise ValueError(f"CSV line {reader.line_num}: expected {len(headers)} cells, got {len(row)}.")
                    if any("\x00" in cell for cell in row):
                        raise ValueError("This file contains binary data, not a readable CSV.")
                    count += 1
                    if count > MAX_ROWS:
                        raise ValueError(f"CSV supports at most {MAX_ROWS:,} records.")
                    if count * len(headers) > MAX_CELLS:
                        raise ValueError(f"CSV supports at most {MAX_CELLS:,} data cells (rows × columns).")
                    for column, value in zip(columns, row):
                        column.append(value)
            except csv.Error as error:
                raise ValueError(f"Malformed CSV near line {reader.line_num}: {error}") from error
            if not count:
                raise ValueError("CSV needs a header and at least one data row.")
            return pd.DataFrame(dict(zip(headers, columns)), dtype=object)
    except UnicodeError as error:
        raise ValueError("Save your CSV as UTF-8 or UTF-16 with a byte-order mark.") from error
    finally:
        binary.close()


def preview(df: pd.DataFrame, limit: int = 5) -> dict:
    return {"columns": list(df.columns), "rows": df.head(limit).fillna("").astype(str).values.tolist()}
