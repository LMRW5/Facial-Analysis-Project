from insightface.model_zoo import get_model
import numpy as np
import cv2
from pathlib import Path

class FaceEmbedder:
    def __init__(self):
        self.model = get_model(
            "models/w600k_r50.onnx",
            providers=["CPUExecutionProvider"]
        )

        self.model.prepare(
            ctx_id=0
        )


    def get_embedding(self, image_path):

        image_path = Path(image_path)

        img = cv2.imread(
            str(image_path)
        )

        if img is None:
            print("Image not found")
            return None


        img = cv2.resize(
            img,
            (112,112)
        )


        embedding = self.model.get_feat(
            img
        )


        return embedding.flatten().astype(
            np.float32
        )

    