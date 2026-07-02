#!/usr/bin/env python3
"""Download a Roboflow classification dataset into folder layout for transfer_probe."""
from __future__ import annotations

import argparse
import os
import zipfile
from pathlib import Path

import urllib.request


def download_and_extract(url: str, dest_dir: Path) -> Path:
    """Download a zip export and extract it under ``dest_dir``."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    zip_path = dest_dir / "roboflow_export.zip"
    print(f"Downloading {url} ...")
    urllib.request.urlretrieve(url, zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest_dir)
    zip_path.unlink(missing_ok=True)
    return dest_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch a Roboflow dataset zip export")
    parser.add_argument(
        "--url",
        required=False,
        help="Roboflow dataset zip URL (set ROBOFLOW_EXPORT_URL env var as fallback)",
    )
    parser.add_argument("--out-dir", default="data/transfer")
    args = parser.parse_args()

    url = args.url or os.environ.get("ROBOFLOW_EXPORT_URL")
    if not url:
        raise SystemExit(
            "Provide --url or set ROBOFLOW_EXPORT_URL to a Roboflow zip export link.\n"
            "Expected layout after extract: train/<class>/*.jpg and val/<class>/*.jpg"
        )

    out = download_and_extract(url, Path(args.out_dir))
    print(f"Dataset ready under {out}")
    print("Run: python scripts/transfer_roboflow.py --data-dir data/transfer")


if __name__ == "__main__":
    main()
