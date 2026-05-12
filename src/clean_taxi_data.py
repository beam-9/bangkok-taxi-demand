"""Clean OTP taxi analytics data for monthly modeling."""

from __future__ import annotations

import re
import sys
import unicodedata
from pathlib import Path

import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import CLEAN_TAXI_PATH, RAW_TAXI_ANALYTICS_PATH, TARGET_PRIORITY, ensure_directories  # noqa: E402


COLUMN_MAP = {
    "เดือน": "date",
    "ปี": "year",
    "พ.ศ.": "year",
    "ปี พ.ศ.": "year",
    "จำนวนเที่ยวรับผู้โดยสาร": "passenger_pickup_trips",
    "จำนวนเที่ยวรับผู้โดยสารเฉลี่ยต่อวัน": "avg_trips_per_day",
    "จำนวนเที่ยวเฉลี่ยต่อวัน": "avg_trips_per_day",
    "ระยะเวลาการเดินทางรวม": "total_passenger_travel_time",
    "เวลาเดินทางรวม": "total_passenger_travel_time",
    "ระยะทางรวม": "total_distance",
    "จำนวนรถแท็กซี่เฉลี่ยต่อเดือน": "average_taxis_per_month",
    "จำนวนรถแท็กซี่เฉลี่ย": "average_taxis_per_month",
}

MONTHS_TH = {
    "มกราคม": 1,
    "กุมภาพันธ์": 2,
    "มีนาคม": 3,
    "เมษายน": 4,
    "พฤษภาคม": 5,
    "มิถุนายน": 6,
    "กรกฎาคม": 7,
    "สิงหาคม": 8,
    "กันยายน": 9,
    "ตุลาคม": 10,
    "พฤศจิกายน": 11,
    "ธันวาคม": 12,
    "ม.ค.": 1,
    "ก.พ.": 2,
    "มี.ค.": 3,
    "เม.ย.": 4,
    "พ.ค.": 5,
    "มิ.ย.": 6,
    "ก.ค.": 7,
    "ส.ค.": 8,
    "ก.ย.": 9,
    "ต.ค.": 10,
    "พ.ย.": 11,
    "ธ.ค.": 12,
}


def snake_case(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", str(value)).strip().lower()
    if normalized in COLUMN_MAP:
        return COLUMN_MAP[normalized]
    for thai, english in COLUMN_MAP.items():
        if thai in normalized:
            return english
    normalized = re.sub(r"[\s\-/()]+", "_", normalized)
    normalized = re.sub(r"[^0-9a-zA-Z_\u0E00-\u0E7F]+", "", normalized)
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    return normalized or "unnamed_column"


def read_taxi_file(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run python src/download_data.py first.")
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    for encoding in ["utf-8-sig", "utf-8", "cp874"]:
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path)


def parse_buddhist_year(year: float | int | str) -> int | None:
    if pd.isna(year):
        return None
    match = re.search(r"\d{4}", str(year))
    if not match:
        return None
    parsed = int(match.group())
    return parsed - 543 if parsed > 2400 else parsed


def parse_month_value(value: object) -> int | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    if text in MONTHS_TH:
        return MONTHS_TH[text]
    for month_name, month_number in MONTHS_TH.items():
        if month_name in text:
            return month_number
    match = re.search(r"(?<!\d)(1[0-2]|0?[1-9])(?!\d)", text)
    return int(match.group()) if match else None


def build_date(df: pd.DataFrame) -> pd.Series:
    lower_cols = {col.lower(): col for col in df.columns}
    if "date" in df.columns:
        parsed = pd.to_datetime(df["date"], errors="coerce")
        if parsed.notna().any():
            return parsed.dt.to_period("M").dt.to_timestamp()

    month_col = next((col for col in df.columns if col in {"month", "เดือน"} or "month" in col.lower()), None)
    year_col = next((col for col in df.columns if col in {"year", "ปี"} or "year" in col.lower()), None)
    if month_col and year_col:
        years = df[year_col].map(parse_buddhist_year)
        months = df[month_col].map(parse_month_value)
        return pd.to_datetime(
            {"year": years, "month": months, "day": 1},
            errors="coerce",
        )

    for col in lower_cols.values():
        parsed = pd.to_datetime(df[col], errors="coerce")
        if parsed.notna().mean() > 0.5:
            return parsed.dt.to_period("M").dt.to_timestamp()

    raise ValueError("Could not identify a usable month/date column in the taxi analytics file.")


def choose_target(columns: list[str]) -> str:
    for target in TARGET_PRIORITY:
        if target in columns:
            return target
    numeric_like = [col for col in columns if col not in {"date", "year", "month"}]
    if not numeric_like:
        raise ValueError("No target candidate found after cleaning taxi analytics columns.")
    return numeric_like[0]


def clean_taxi_data(input_path: Path = RAW_TAXI_ANALYTICS_PATH, output_path: Path = CLEAN_TAXI_PATH) -> pd.DataFrame:
    df = read_taxi_file(input_path)
    print("Original columns:")
    for col in df.columns:
        print(f"- {col}")

    df = df.rename(columns={col: snake_case(col) for col in df.columns})
    df["date"] = build_date(df)
    df = df.dropna(subset=["date"]).copy()
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month

    for col in df.columns:
        if col != "date":
            converted = pd.to_numeric(df[col].astype(str).str.replace(",", "", regex=False), errors="coerce")
            if converted.notna().sum() >= df[col].notna().sum() * 0.5:
                df[col] = converted

    target = choose_target(list(df.columns))
    df["target"] = pd.to_numeric(df[target], errors="coerce")
    df = df.sort_values("date").drop_duplicates(subset=["date"], keep="last")

    ensure_directories()
    df.to_csv(output_path, index=False)
    print(f"Selected target column: {target}")
    print(f"Saved {output_path}")
    return df


if __name__ == "__main__":
    clean_taxi_data()

