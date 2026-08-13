import os
import json
import faiss
import numpy as np

from tools.objects.faceEmbedder import FaceEmbedder

# -------------------------
# SETTINGS
# -------------------------

DATASET_DIR = "Open Famous People Faces"
DATABASE_DIR = "database"

MIN_IMAGES_PER_PERSON = 2


# -------------------------
# BUILD DATABASE
# -------------------------

def main():
    os.makedirs(DATABASE_DIR, exist_ok=True)

    embedder = FaceEmbedder()

    vectors = []
    metadata = []
    id_counter = 0

    print("Starting database generation...\n")

    # Traverse each person's folder in the dataset
    for person_name in os.listdir(DATASET_DIR):
        person_folder = os.path.join(DATASET_DIR, person_name)

        if not os.path.isdir(person_folder):
            continue

        # Get valid image files
        image_names = [
            name
            for name in os.listdir(person_folder)
            if name.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        if len(image_names) < MIN_IMAGES_PER_PERSON:
            continue

        print(f"Processing Person: {person_name} ({len(image_names)} images)")

        person_embeddings = []

        # Extract embeddings for all images of this person
        for image_name in image_names:
            image_path = os.path.join(person_folder, image_name)

            embedding = embedder.get_embedding(image_path)

            if embedding is None:
                print(f"  └─ Failed to extract embedding for {image_name}")
                continue

            person_embeddings.append(embedding)

        # Skip if we couldn't get enough valid embeddings
        if len(person_embeddings) < MIN_IMAGES_PER_PERSON:
            print(f"  └─ Skipped: Not enough valid face embeddings extracted.\n")
            continue

        # ---------------------------------------------------------
        # AVERAGE VECTOR COMPUTATION (Look-Alike Composite Vector)
        # ---------------------------------------------------------
        # 1. Compute the mean feature vector across all photos of this person
        mean_vector = np.mean(person_embeddings, axis=0)

        # 2. Normalize the composite mean vector to unit length (L2 norm = 1)
        norm = np.linalg.norm(mean_vector)
        if norm > 0:
            mean_vector = mean_vector / norm

        vectors.append(mean_vector)

        # 3. Store single composite metadata entry per celebrity
        metadata.append({
            "id": id_counter,
            "person": person_name,
            "sample_count": len(person_embeddings)
        })

        id_counter += 1
        print(f"  └─ Successfully created composite vector (ID: {id_counter - 1})\n")

    # -------------------------
    # CHECK DATA
    # -------------------------

    print("--------------------------------------------------")
    print(f"Total Unique Celebrities Processed: {len(vectors)}")

    if len(vectors) == 0:
        print("No valid face vectors were generated. Exiting.")
        return

    # Convert vectors list to a 2D float32 numpy array
    vectors_array = np.array(vectors, dtype="float32")

    print(f"FAISS Matrix Shape: {vectors_array.shape}")

    # Double-check L2 normalization for Inner Product (Cosine Similarity) index
    faiss.normalize_L2(vectors_array)

    # -------------------------
    # CREATE FAISS INDEX
    # -------------------------

    dimension = vectors_array.shape[1]  # 512 for ArcFace

    # IndexFlatIP calculates exact Cosine Similarity on L2-normalized vectors
    index = faiss.IndexFlatIP(dimension)
    index.add(vectors_array)

    print(f"FAISS Total Indexed Vectors: {index.ntotal}")

    # -------------------------
    # SAVE FAISS INDEX & METADATA
    # -------------------------

    index_path = os.path.join(DATABASE_DIR, "faces.faiss")
    metadata_path = os.path.join(DATABASE_DIR, "metadata.json")

    faiss.write_index(index, index_path)

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)

    print("--------------------------------------------------")
    print("Database build complete!")
    print(f"Saved FAISS Index : {index_path}")
    print(f"Saved Metadata    : {metadata_path}")


# -------------------------
# RUN
# -------------------------

if __name__ == "__main__":
    main()