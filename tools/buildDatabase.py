import sys
from pathlib import Path

import faiss
import numpy as np

from tools.processing.embedding_converter import get_embedding


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = PROJECT_ROOT / "dataset"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def parse_metadata(phenotype_dir: Path, image_path: Path) -> dict:
    stem = image_path.stem
    cropped = stem.endswith("_cropped")
    if cropped:
        stem = stem[: -len("_cropped")]

    low_res = stem.endswith("_low_res")
    if low_res:
        stem = stem[: -len("_low_res")]

    gender = "female" if stem.endswith("f") else "male" if stem.endswith("m") else "unknown"

    return {
        "phenotype": phenotype_dir.name,
        "gender": gender,
        "resolution": "low_res" if low_res else "regular",
        "file": image_path.name,
        "cropped": cropped,
    }


embeddings = []
metadata = []

if not DATASET_PATH.exists() or not DATASET_PATH.is_dir():
    print(f"Dataset folder not found: {DATASET_PATH}", file=sys.stderr)
    sys.exit(1)

for phenotype_dir in sorted(DATASET_PATH.iterdir()):
    if not phenotype_dir.is_dir():
        continue

    for image_path in sorted(phenotype_dir.iterdir()):
        if not is_image_file(image_path):
            continue

        print("Processing:", image_path)
        vector = get_embedding(str(image_path))
        if vector is None:
            continue

        embeddings.append(vector)
        metadata.append(parse_metadata(phenotype_dir, image_path))

print("Total faces:", len(embeddings))

embeddings = np.array(embeddings, dtype="float32")
print(embeddings.shape)

# Optional: build a FAISS index for similarity search.
if embeddings.size > 0:
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    index_path = PROJECT_ROOT / "database" / "face_index.faiss"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_path))
    print(f"Saved FAISS index to: {index_path}")