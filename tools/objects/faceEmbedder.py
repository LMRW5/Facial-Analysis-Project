import cv2
import numpy as np
from pathlib import Path
from insightface.model_zoo import get_model

class FaceEmbedder:
    def __init__(self):
        # Load the ArcFace model directly (No face detector needed!)
        self.model = get_model(
            "models/w600k_r50.onnx",
            providers=["CPUExecutionProvider"]
        )
        self.model.prepare(ctx_id=0)

    def _letterbox_preprocess(self, img, target_size=(112, 112)):
        """
        Pads rectangular face crops into a square with neutral gray padding 
        and resizes to 112x112 without stretching facial proportions.
        """
        h, w = img.shape[:2]
        target_w, target_h = target_size

        # Determine scaling factor to preserve aspect ratio
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)

        # Resize image cleanly
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Create a neutral gray canvas (128, 128, 128)
        canvas = np.full((target_h, target_w, 3), 128, dtype=np.uint8)

        # Center the resized face crop inside the canvas
        top = (target_h - new_h) // 2
        left = (target_w - new_w) // 2
        canvas[top:top + new_h, left:left + new_w] = resized

        return canvas

    def get_embedding(self, image_path):
        image_path = Path(image_path)
        img = cv2.imread(str(image_path))

        if img is None:
            print(f"Image not found: {image_path}")
            return None

        # 1. Convert BGR (OpenCV default) to RGB (ArcFace requirement)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 2. Pad & resize to 112x112 without distorting aspect ratio
        aligned_crop = self._letterbox_preprocess(img_rgb, target_size=(112, 112))

        # 3. Extract features via ArcFace
        embedding = self.model.get_feat(aligned_crop).flatten().astype(np.float32)

        # 4. L2 Normalize vector for FAISS Cosine Similarity
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding