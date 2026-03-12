"""
setup_data.py – Download and unpack the RVL-CDIP Small dataset from Kaggle.

Prerequisites
-------------
1. Install the Kaggle API:
       pip install kaggle

2. Create a Kaggle API token:
   - Go to https://www.kaggle.com/settings  →  "API"  →  "Create New Token"
   - This downloads  kaggle.json
   - On Windows, place it at:  C:\\Users\\<you>\\.kaggle\\kaggle.json

Usage
-----
  python setup_data.py

What it does
------------
  1. Downloads  uditamin/rvl-cdip-small  via the Kaggle API
  2. Unzips it into  data/rvl-cdip-small/
  3. Verifies that  images/  and  labels/  folders are present
  4. Prints a summary of available label files and image count
"""

import os
import sys
import zipfile
from pathlib import Path

DATASET_SLUG = "uditamin/rvl-cdip-small"
DEST_DIR     = Path("data/rvl-cdip-small")
ZIP_NAME     = "rvl-cdip-small.zip"


def check_kaggle_credentials() -> bool:
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if kaggle_json.exists():
        return True
    env_set = os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY")
    return bool(env_set)


def download_dataset() -> None:
    try:
        import kaggle  # noqa: F401 – triggers credential check
    except ImportError:
        print("  ERROR: Kaggle package not installed.  Run:  pip install kaggle")
        sys.exit(1)

    if not check_kaggle_credentials():
        print(
            "\n  ERROR: Kaggle credentials not found.\n"
            "  1. Go to https://www.kaggle.com/settings → API → Create New Token\n"
            "  2. Save kaggle.json to  C:\\Users\\<you>\\.kaggle\\kaggle.json\n"
        )
        sys.exit(1)

    print(f"  Downloading dataset: {DATASET_SLUG} …")
    import kaggle.api as kapi
    kapi.authenticate()
    kapi.dataset_download_files(
        DATASET_SLUG,
        path=str(Path("data")),
        unzip=False,
        quiet=False,
    )


def unzip_dataset() -> None:
    zip_path = Path("data") / ZIP_NAME
    if not zip_path.exists():
        # Kaggle may name it differently; try to find any zip in data/
        zips = list(Path("data").glob("*.zip"))
        if not zips:
            print("  ERROR: No zip file found in data/ after download.")
            sys.exit(1)
        zip_path = zips[0]

    print(f"  Extracting {zip_path.name} → {DEST_DIR} …")
    DEST_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(DEST_DIR)
    zip_path.unlink()
    print("  Zip removed after extraction.")


def verify_structure() -> None:
    print("\n  Verifying dataset structure …")
    issues: list[str] = []

    images_dir = DEST_DIR / "images"
    labels_dir = DEST_DIR / "labels"

    if not images_dir.exists():
        issues.append(f"  ✗ Missing:  {images_dir}")
    if not labels_dir.exists():
        issues.append(f"  ✗ Missing:  {labels_dir}")

    if issues:
        print("\n".join(issues))
        print(
            "\n  The zip may have a different internal layout.\n"
            f"  Please move the contents so that {DEST_DIR}/images/ and\n"
            f"  {DEST_DIR}/labels/ exist, then re-run.\n"
        )
        sys.exit(1)

    # Count images and summarise label files
    image_count = sum(1 for _ in images_dir.rglob("*.tif"))
    print(f"  ✓ images/   found  ({image_count:,} .tif files)")

    for split in ("train", "val", "test"):
        lf = labels_dir / f"{split}.txt"
        if lf.exists():
            lines = lf.read_text(encoding="utf-8").strip().splitlines()
            print(f"  ✓ labels/{split}.txt  ({len(lines):,} entries)")
        else:
            print(f"  – labels/{split}.txt  not found (optional)")


def main() -> None:
    print("\n" + "=" * 52)
    print("  SETUP DATA  –  RVL-CDIP Small (Kaggle)")
    print("=" * 52 + "\n")

    if (DEST_DIR / "labels").exists():
        print(f"  Dataset already present at {DEST_DIR}")
        verify_structure()
        print("\n  Nothing to do.  Run  python train.py  to start training.\n")
        return

    download_dataset()
    unzip_dataset()
    verify_structure()

    print(
        "\n  Dataset ready.\n"
        "  Next steps:\n"
        "    Quick test (first 100 images/class):\n"
        "      python train.py --limit 100\n"
        "    Full training:\n"
        "      python train.py\n"
    )


if __name__ == "__main__":
    main()
