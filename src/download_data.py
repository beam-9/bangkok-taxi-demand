"""Download public taxi mobility and Bangkok weather data."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import requests

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import (  # noqa: E402
    RAW_TAXI_ANALYTICS_PATH,
    RAW_TAXI_HOTSPOTS_PATH,
    RAW_WEATHER_PATH,
    TAXI_ANALYTICS_PAGE,
    TAXI_ANALYTICS_URL,
    TAXI_HOTSPOTS_PAGE,
    TAXI_HOTSPOTS_URL,
    WEATHER_URL,
    ensure_directories,
)


def download_file(url: str, output_path: Path, dataset_page: str | None = None) -> None:
    """Download a URL to disk with a clear manual fallback message."""
    try:
        response = requests.get(url, timeout=60)
        response.raise_for_status()
    except requests.RequestException as exc:
        message = f"Failed to download {url}\nReason: {exc}"
        if dataset_page:
            message += f"\nPlease manually download the file from {dataset_page} and save it as {output_path}."
        raise RuntimeError(message) from exc

    if len(response.content) < 100:
        raise RuntimeError(f"Downloaded file from {url} looks too small. Please verify the source.")

    output_path.write_bytes(response.content)
    print(f"Saved {output_path}")


def download_weather(output_path: Path) -> None:
    """Download Open-Meteo hourly weather JSON and save as CSV."""
    try:
        response = requests.get(WEATHER_URL, timeout=60)
        response.raise_for_status()
        payload = response.json()
    except requests.RequestException as exc:
        raise RuntimeError(f"Failed to download Open-Meteo weather data: {exc}") from exc

    hourly = payload.get("hourly")
    if not hourly:
        raise RuntimeError("Open-Meteo response did not contain an 'hourly' section.")

    pd.DataFrame(hourly).to_csv(output_path, index=False)
    print(f"Saved {output_path}")


def main() -> None:
    ensure_directories()
    download_file(TAXI_ANALYTICS_URL, RAW_TAXI_ANALYTICS_PATH, TAXI_ANALYTICS_PAGE)
    download_file(TAXI_HOTSPOTS_URL, RAW_TAXI_HOTSPOTS_PATH, TAXI_HOTSPOTS_PAGE)
    download_weather(RAW_WEATHER_PATH)


if __name__ == "__main__":
    main()

