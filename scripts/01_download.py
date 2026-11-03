"""Download the M5 dataset (GitHub mirror of the Kaggle files) into data/raw."""
import sys
import urllib.request
import zipfile
from pathlib import Path

URL = "https://github.com/Nixtla/m5-forecasts/raw/main/datasets/m5.zip"
RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
NEEDED = ["sales_train_evaluation.csv", "sales_test_evaluation.csv", "sell_prices.csv", "calendar.csv"]


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    if all((RAW / f).exists() for f in NEEDED):
        print("already downloaded")
        return
    archive = RAW / "m5.zip"
    if not archive.exists():
        print(f"downloading {URL}")
        req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=120) as r, open(archive, "wb") as out:
            while chunk := r.read(1 << 20):
                out.write(chunk)
    with zipfile.ZipFile(archive) as z:
        for name in NEEDED:
            z.extract(name, RAW)
    archive.unlink()
    for f in NEEDED:
        print(f, (RAW / f).stat().st_size)


if __name__ == "__main__":
    sys.exit(main())
