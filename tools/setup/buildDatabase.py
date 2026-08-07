import os
import json
import faiss
import numpy as np

from faceEmbedder import FaceEmbedder


DATASET_DIR = "dataset"
DATABASE_DIR = "database"

def main():
    os.makedirs(
        DATABASE_DIR,
        exist_ok=True
    )


    embedder = FaceEmbedder()


    vectors = []
    metadata = []


    id_counter = 0


    for phenotype_folder in os.listdir(DATASET_DIR):

        folder_path = os.path.join(
            DATASET_DIR,
            phenotype_folder
        )


        if not os.path.isdir(folder_path):
            continue


        for image_name in os.listdir(folder_path):
            if not image_name.lower().endswith(
                (".jpg", ".png", ".jpeg")
            ) or not "cropped" in image_name or "low_res" in image_name:
                continue
            
            parts = image_name.split("_")
            gender = None

            if len(parts) > 1:
                gender = parts[0][-1]


            image_path = os.path.join(
                folder_path,
                image_name
            )

            print(
                "Processing:",
                image_path
            )


            embedding = embedder.get_embedding(
                image_path
            )


            if embedding is None:
                continue


            vectors.append(
                embedding
            )


            metadata.append(
                {
                    "id": id_counter,
                    "phenotype": phenotype_folder,
                    "file": image_name,
                    "gender": gender
                }
            )


            id_counter += 1



    print(
        "Total embeddings:",
        len(vectors)
    )

    if len(vectors) == 0:
        print("No images found!")
        return
    # Convert to numpy matrix

    vectors = np.array(
        vectors,
        dtype="float32"
    )


    print(
        "Vector shape:",
        vectors.shape
    )


    # Normalize for cosine similarity

    faiss.normalize_L2(
        vectors
    )


    # Create FAISS index

    dimension = 512

    index = faiss.IndexFlatIP(
        dimension
    )


    index.add(
        vectors
    )

    print(
        "FAISS vectors:",
        index.ntotal
    )


    # Save FAISS database

    faiss.write_index(
        index,
        f"{DATABASE_DIR}/phenotypes.faiss"
    )


    # Save metadata

    with open(
        f"{DATABASE_DIR}/metadata.json",
        "w"
    ) as f:

        json.dump(
            metadata,
            f,
            indent=4
        )


    print(
        "Database created!"
    )
if __name__ == "__main__":
    main()