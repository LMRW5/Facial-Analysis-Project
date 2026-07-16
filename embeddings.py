import cv2
import numpy as np
from insightface.app import FaceAnalysis


app = FaceAnalysis(
    name="buffalo_l"
)

app.prepare(
    ctx_id=0,
    det_size=(640, 640)
)


def get_embedding(image_path):

    img = cv2.imread(image_path)

    if img is None:
        print("Could not load:", image_path)
        return None


    # Resize because ArcFace expects a consistent face input
    img = cv2.resize(
        img,
        (112,112)
    )


    faces = app.get(img)


    if len(faces) == 0:
        print("No face detected:", image_path)
        return None


    embedding = faces[0].embedding


    return embedding.astype(
        np.float32
    )