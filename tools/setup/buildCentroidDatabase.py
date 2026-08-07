import os
import json
import faiss
import numpy as np

from tools.setup.faceEmbedder import FaceEmbedder


DATASET_DIR = "dataset"
DATABASE_DIR = "database"


def main():

    os.makedirs(
        DATABASE_DIR,
        exist_ok=True
    )


    embedder = FaceEmbedder()


    # Stores:
    # {
    #   "southmongolid": [
    #       embedding1,
    #       embedding2
    #   ]
    # }

    phenotype_embeddings = {}


    for phenotype_folder in os.listdir(DATASET_DIR):

        folder_path = os.path.join(
            DATASET_DIR,
            phenotype_folder
        )


        if not os.path.isdir(folder_path):
            continue


        for image_name in os.listdir(folder_path):

            if (
                not image_name.lower().endswith(
                    (".jpg",".png",".jpeg")
                )
                or "cropped" not in image_name
                or "low_res" in image_name
            ):
                continue


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


            if phenotype_folder not in phenotype_embeddings:
                phenotype_embeddings[phenotype_folder] = []


            phenotype_embeddings[phenotype_folder].append(
                embedding
            )


    centroid_vectors = []
    centroid_metadata = []


    id_counter = 0


    for phenotype, embeddings in phenotype_embeddings.items():


        embeddings = np.array(
            embeddings,
            dtype="float32"
        )


        # Average all images of this phenotype
        centroid = np.mean(
            embeddings,
            axis=0
        )


        centroid_vectors.append(
            centroid
        )


        centroid_metadata.append(
            {
                "id": id_counter,
                "phenotype": phenotype,
                "images_used": len(embeddings)
            }
        )


        id_counter += 1



    print(
        "Total phenotypes:",
        len(centroid_vectors)
    )


    centroid_vectors = np.array(
        centroid_vectors,
        dtype="float32"
    )


    print(
        "Centroid shape:",
        centroid_vectors.shape
    )


    faiss.normalize_L2(
        centroid_vectors
    )


    index = faiss.IndexFlatIP(
        512
    )


    index.add(
        centroid_vectors
    )


    print(
        "FAISS centroids:",
        index.ntotal
    )


    faiss.write_index(
        index,
        f"{DATABASE_DIR}/phenotype_centroids.faiss"
    )


    with open(
        f"{DATABASE_DIR}/centroid_metadata.json",
        "w"
    ) as f:

        json.dump(
            centroid_metadata,
            f,
            indent=4
        )


    print(
        "Centroid database created!" 
    )


if __name__ == "__main__":
    main()