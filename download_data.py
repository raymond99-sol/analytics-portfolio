"""Download the two public UCI datasets used by the portfolio notebooks."""

from __future__ import annotations

import argparse
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RETAIL_URL = "https://archive.ics.uci.edu/static/public/352/online%2Bretail.zip"
BANK_URL = "https://archive.ics.uci.edu/static/public/222/bank%2Bmarketing.zip"


def download(url: str, destination: Path) -> None:
    print(f"Downloading {url}")
    with urllib.request.urlopen(url) as response, destination.open("wb") as output:
        shutil.copyfileobj(response, output)


def prepare_retail(temp_dir: Path) -> None:
    archive = temp_dir / "online_retail.zip"
    download(RETAIL_URL, archive)
    destination = ROOT / "project_1_retail_customer_segmentation" / "data"
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zipped:
        zipped.extract("Online Retail.xlsx", destination)


def prepare_bank(temp_dir: Path) -> None:
    outer_archive = temp_dir / "bank_marketing.zip"
    download(BANK_URL, outer_archive)
    with zipfile.ZipFile(outer_archive) as zipped:
        zipped.extract("bank-additional.zip", temp_dir)

    destination = ROOT / "project_2_bank_campaign_optimization" / "data"
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(temp_dir / "bank-additional.zip") as zipped:
        for member in [
            "bank-additional/bank-additional-full.csv",
            "bank-additional/bank-additional-names.txt",
        ]:
            extracted = Path(zipped.extract(member, temp_dir))
            shutil.copy2(extracted, destination / extracted.name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download public source data for one or both portfolio projects."
    )
    parser.add_argument(
        "--project",
        choices=("retail", "bank", "all"),
        default="all",
        help="Dataset to download (default: all).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with tempfile.TemporaryDirectory() as temp:
        temp_dir = Path(temp)
        if args.project in {"retail", "all"}:
            prepare_retail(temp_dir)
        if args.project in {"bank", "all"}:
            prepare_bank(temp_dir)
    print(f"Data download completed for: {args.project}.")


if __name__ == "__main__":
    main()
