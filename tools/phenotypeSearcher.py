import faiss
import json
import numpy as np

from faceEmbedder import FaceEmbedder


INDEX_PATH = "database/phenotypes.faiss"
METADATA_PATH = "database/metadata.json"


class PhenotypeSearcher:

    def __init__(self):

        self.index = faiss.read_index(
            INDEX_PATH
        )

        with open(METADATA_PATH, "r") as f:
            self.metadata = json.load(f)

        self.embedder = FaceEmbedder()


    def search(self, image_path, k=100):

        embedding = self.embedder.get_embedding(
            image_path
        )

        if embedding is None:
            return []


        query = np.array(
            [embedding],
            dtype="float32"
        )


        faiss.normalize_L2(
            query
        )


        distances, ids = self.index.search(
            query,
            k
        )


        results = []


        for score, idx in zip(
            distances[0],
            ids[0]
        ):

            results.append(
                {
                    "score": float(score),
                    **self.metadata[idx]
                }
            )


        return results[:10:]