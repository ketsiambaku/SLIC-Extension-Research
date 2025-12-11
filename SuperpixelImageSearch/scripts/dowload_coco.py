import os
import zipfile
from urllib.request import urlretrieve

# ROOT = SLIC-Extension-Research/SuperpixelImageSearch
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

DATA = os.path.join(ROOT, "data", "coco2017")
IMG  = os.path.join(DATA, "images")
ANN  = os.path.join(DATA, "annotations")

MAX_IMAGES = 2500  # demo limit

URLS = {
    "train2017.zip": "http://images.cocodataset.org/zips/train2017.zip",
    "val2017.zip": "http://images.cocodataset.org/zips/val2017.zip",
    "annotations_trainval2017.zip":
        "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
}

def download(url, dst):
    print(f"Downloading {url} → {dst}")
    urlretrieve(url, dst)

def unzip_limited_images(zip_path, dst, max_images):
    print(f"Extracting up to {max_images} images from {zip_path}")
    os.makedirs(dst, exist_ok=True)

    extracted = 0
    with zipfile.ZipFile(zip_path, "r") as zf:
        for name in zf.namelist():
            if not name.lower().endswith(".jpg"):
                continue
            zf.extract(name, dst)
            extracted += 1
            if extracted >= max_images:
                break

    print(f"Extracted {extracted} images.")

def unzip_all(zip_path, dst):
    print(f"Extracting {zip_path}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dst)

def safe_delete(path):
    """Delete a file if it exists."""
    if os.path.exists(path):
        print(f"Deleting {path}...")
        os.remove(path)
    else:
        print(f"{path} not found — skipping.")

def main():
    os.makedirs(IMG, exist_ok=True)
    os.makedirs(ANN, exist_ok=True)

    # Download ZIPs
    for filename, url in URLS.items():
        dst = os.path.join(DATA, filename)
        if not os.path.exists(dst):
            download(url, dst)
        else:
            print(f"{filename} already exists.")

    # Extract limited training images
    unzip_limited_images(
        os.path.join(DATA, "train2017.zip"),
        IMG,
        MAX_IMAGES
    )

    # Extract full validation set
    unzip_all(
        os.path.join(DATA, "val2017.zip"),
        IMG
    )

    # Extract annotations
    unzip_all(
        os.path.join(DATA, "annotations_trainval2017.zip"),
        ANN
    )

    print("Extraction complete — cleaning up ZIP files...")

    # --- DELETE ZIP FILES ---
    safe_delete(os.path.join(DATA, "train2017.zip"))
    safe_delete(os.path.join(DATA, "val2017.zip"))
    safe_delete(os.path.join(DATA, "annotations_trainval2017.zip"))
    # -------------------------

    print("✅ COCO demo dataset ready in SuperpixelImageSearch/data")
    print("🧹 All ZIP files removed.")

if __name__ == "__main__":
    main()
