import faiss
import json
import numpy as np

from server.ML.tools.objects.faceEmbedder import FaceEmbedder


INDEX_PATH = "database/faces.faiss"
METADATA_PATH = "database/metadata.json"


class FaceSearcher:

    def __init__(self):

        self.index = faiss.read_index(
            INDEX_PATH
        )

        with open(
            METADATA_PATH,
            "r"
        ) as f:

            self.metadata = json.load(f)


        self.embedder = FaceEmbedder()


    def search(
        self,
        image_path,
        top_k=10
    ):

        embedding = self.embedder.get_embedding(
            image_path
        )


        if embedding is None:
            return []


        query = np.array(
            [embedding],
            dtype="float32"
        )


        # Normalize for cosine similarity
        faiss.normalize_L2(
            query
        )


        # Search FAISS
        distances, ids = self.index.search(
            query,
            top_k
        )


        results = []


        for score, idx in zip(
            distances[0],
            ids[0]
        ):

            if idx < 0:
                continue


            results.append(
                {
                    "score": float(score),
                    **self.metadata[idx]
                }
            )


        return results