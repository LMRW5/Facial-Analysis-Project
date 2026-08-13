import os
import random
import faiss
import cv2
import numpy as np

from pathlib import Path

from tools.objects.faceEmbedder import FaceEmbedder


# ============================================================
# SETTINGS
# ============================================================

DATASET_DIR = Path("Open Famous People Faces")

REFERENCE_PERCENT = 0.80

TOP_K = 5

RANDOM_SEED = 42

MIN_IMAGES_PER_PERSON = 5


# ============================================================
# GET IMAGES
# ============================================================

def get_images(person_dir):

    valid_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".webp"
    }

    return [
        path
        for path in person_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() in valid_extensions
    ]


# ============================================================
# MAIN
# ============================================================

def main():

    random.seed(
        RANDOM_SEED
    )


    embedder = FaceEmbedder()


    reference_vectors = []

    reference_people = []

    test_images = []


    # ========================================================
    # BUILD REFERENCE / TEST SPLIT
    # ========================================================

    people = [
        folder
        for folder in DATASET_DIR.iterdir()
        if folder.is_dir()
    ]


    print(
        f"People found: {len(people)}"
    )


    for person_dir in people:

        person = person_dir.name

        images = get_images(
            person_dir
        )


        if len(images) < MIN_IMAGES_PER_PERSON:
            continue


        random.shuffle(
            images
        )


        split_index = max(
            1,
            int(
                len(images)
                * REFERENCE_PERCENT
            )
        )


        reference_images = images[
            :split_index
        ]

        test_person_images = images[
            split_index:
        ]


        # Make sure at least one test image exists
        if not test_person_images:
            continue


        # ====================================================
        # EMBED REFERENCE IMAGES
        # ====================================================

        person_vectors = []


        for image_path in reference_images:

            embedding = embedder.get_embedding(
                str(image_path)
            )


            if embedding is None:
                continue


            vector = np.asarray(
                embedding,
                dtype="float32"
            )


            faiss.normalize_L2(
                vector.reshape(1, -1)
            )


            person_vectors.append(
                vector
            )


        if not person_vectors:
            continue


        # ====================================================
        # CREATE PERSON CENTROID
        # ====================================================

        centroid = np.mean(
            np.array(
                person_vectors,
                dtype="float32"
            ),
            axis=0
        )


        # Normalize centroid
        centroid = centroid.reshape(
            1, -1
        )

        faiss.normalize_L2(
            centroid
        )


        reference_vectors.append(
            centroid[0]
        )

        reference_people.append(
            person
        )


        # ====================================================
        # STORE TEST IMAGES
        # ====================================================

        for image_path in test_person_images:

            test_images.append(
                {
                    "person": person,
                    "path": image_path
                }
            )


    # ========================================================
    # CHECK DATA
    # ========================================================

    if not reference_vectors:

        print(
            "No reference data available."
        )

        return


    print(
        f"Reference people: "
        f"{len(reference_people)}"
    )

    print(
        f"Test images: "
        f"{len(test_images)}"
    )


    # ========================================================
    # CREATE TEMPORARY FAISS INDEX
    # ========================================================

    vectors = np.array(
        reference_vectors,
        dtype="float32"
    )


    dimension = vectors.shape[1]


    index = faiss.IndexFlatIP(
        dimension
    )


    index.add(
        vectors
    )


    # ========================================================
    # EVALUATE
    # ========================================================

    top1_correct = 0

    top5_correct = 0

    total = 0


    print()
    print(
        "Running evaluation..."
    )


    for test in test_images:

        true_person = test["person"]

        image_path = test["path"]


        embedding = embedder.get_embedding(
            str(image_path)
        )


        if embedding is None:
            continue


        query = np.asarray(
            [embedding],
            dtype="float32"
        )


        faiss.normalize_L2(
            query
        )


        scores, ids = index.search(
            query,
            TOP_K
        )


        predicted_people = [
            reference_people[idx]
            for idx in ids[0]
            if idx >= 0
        ]


        if not predicted_people:
            continue


        total += 1


        # Top-1
        if predicted_people[0] == true_person:

            top1_correct += 1


        # Top-5
        if true_person in predicted_people:

            top5_correct += 1


    # ========================================================
    # RESULTS
    # ========================================================

    if total == 0:

        print(
            "No test images could be evaluated."
        )

        return


    top1_accuracy = (
        top1_correct / total
    )

    top5_accuracy = (
        top5_correct / total
    )


    print()
    print("=" * 50)
    print("FACE RECOGNITION EVALUATION")
    print("=" * 50)

    print(
        f"Test images:     {total}"
    )

    print(
        f"Top-1 accuracy:  "
        f"{top1_accuracy * 100:.2f}%"
    )

    print(
        f"Top-5 accuracy:  "
        f"{top5_accuracy * 100:.2f}%"
    )

    print("=" * 50)


if __name__ == "__main__":
    main()