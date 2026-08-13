#This script goes through every image inside of the databse, identifies the face and crops it out

import sys
from pathlib import Path

import cv2

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATASET_DIR = PROJECT_ROOT / "dataset"
CASCADE_PATH = PROJECT_ROOT / "models" / "haarcascade_frontalface_default.xml"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def load_face_detector(cascade_path: Path) -> cv2.CascadeClassifier:
    cascade = cv2.CascadeClassifier(str(cascade_path))
    if cascade.empty():
        raise FileNotFoundError(f"Failed to load Haar cascade: {cascade_path}")
    return cascade


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def crop_face_from_image(img: cv2.Mat, cascade: cv2.CascadeClassifier) -> cv2.Mat | None:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(
        gray,
        scaleFactor=1.05,
        minNeighbors=5,
        minSize=(100, 100),
    )
    if len(faces) == 0:
        return None

    x, y, w, h = max(faces, key=lambda face: face[2] * face[3])
    face = img[y : y + h, x : x + w]
    return cv2.resize(face, (112, 112), interpolation=cv2.INTER_AREA)


def process_dataset(dataset_dir: Path, cascade: cv2.CascadeClassifier) -> int:
    total_saved = 0
    if not dataset_dir.exists() or not dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    for phenotype_dir in sorted(dataset_dir.iterdir()):
        if not phenotype_dir.is_dir():
            continue

        for image_path in sorted(phenotype_dir.iterdir()):
            if not is_image_file(image_path):
                continue
            if image_path.stem.endswith("_cropped"):
                continue

            img = cv2.imread(str(image_path))
            if img is None:
                print(f"Skipping unreadable image: {image_path}")
                continue

            cropped = crop_face_from_image(img, cascade)
            if cropped is None:
                print(f"No face detected in: {image_path}")
                continue

            output_path = image_path.with_name(f"{image_path.stem}_cropped{image_path.suffix}")
            cv2.imwrite(str(output_path), cropped)
            print(f"Saved cropped image: {output_path}")
            total_saved += 1

    return total_saved


def main() -> int:
    try:
        cascade = load_face_detector(CASCADE_PATH)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    try:
        saved_count = process_dataset(DATASET_DIR, cascade)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(f"Finished. Total cropped images saved: {saved_count}")
    return 0


if __name__ == "__main__":
    sys.exit(main())