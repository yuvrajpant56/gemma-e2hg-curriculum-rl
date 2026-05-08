import ast
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


def parse_numbers_field(numbers_raw: Any) -> List[int]:
    """
    Convert different possible number formats into a Python list of ints.

    Supported examples:
      "[82 15]"
      "[82, 15]"
      [82, 15]
      "82 15"
    """
    if isinstance(numbers_raw, list):
        return [int(x) for x in numbers_raw]

    if numbers_raw is None:
        raise ValueError("numbers field is None")

    text = str(numbers_raw).strip()

    if not text:
        return []

    # Case: "[82 15]" or "[82, 15]"
    if text.startswith("[") and text.endswith("]"):
        inner = text[1:-1].strip()
        if not inner:
            return []

        if "," in inner:
            parsed = ast.literal_eval(text)
            return [int(x) for x in parsed]

        return [int(x) for x in inner.split()]

    # Case: "82 15"
    if " " in text and "," not in text:
        return [int(x) for x in text.split()]

    # Case: "82,15"
    if "," in text:
        return [int(x.strip()) for x in text.split(",")]

    raise ValueError(f"Unsupported numbers format: {numbers_raw}")


def _normalize_record(row: Dict[str, Any]) -> Dict[str, Any]:
    # Handle column name differences
    if "nums" in row and "numbers" not in row:
        row["numbers"] = row["nums"]

    required_cols = ["target", "numbers"]
    missing = [col for col in required_cols if col not in row]
    if missing:
        raise ValueError(f"Missing required columns in dataset row: {missing}")

    # Extract optional fields from extra_info if available
    extra = row.get("extra_info", {}) or {}
    if isinstance(extra, str):
        import ast
        extra = ast.literal_eval(extra)

    return {
        "difficulty_n": int(row.get("difficulty_n", len(parse_numbers_field(row["numbers"])))),
        "split": str(row.get("split", extra.get("split", "test"))),
        "sample_idx": int(row.get("sample_idx", extra.get("index", 0))),
        "target": int(row["target"]),
        "numbers": parse_numbers_field(row["numbers"]),
    }


def load_csv_dataset(csv_path: str) -> List[Dict[str, Any]]:
    df = pd.read_csv(csv_path)
    return [_normalize_record(row.to_dict()) for _, row in df.iterrows()]


def load_parquet_dataset(parquet_path: str, max_samples: Optional[int] = None) -> List[Dict[str, Any]]:
    df = pd.read_parquet(parquet_path)

    if max_samples is not None:
        df = df.head(max_samples)

    return [_normalize_record(row.to_dict()) for _, row in df.iterrows()]


def load_dataset(path: str, max_samples: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Automatically load either CSV or parquet based on file extension.
    """
    suffix = Path(path).suffix.lower()

    if suffix == ".csv":
        records = load_csv_dataset(path)
    elif suffix == ".parquet":
        records = load_parquet_dataset(path, max_samples=max_samples)
    else:
        raise ValueError(f"Unsupported dataset file type: {path}")

    if max_samples is not None and suffix == ".csv":
        records = records[:max_samples]

    return records