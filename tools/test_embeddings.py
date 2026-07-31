from faceEmbedder import FaceEmbedder
from buildDatabase import embed_image
embedder = FaceEmbedder()

embedding = embed_image("dataset/aetid/aetidf_cropped.jpg", embedder)

if embedding is not None:
    print("Embedding shape:", embedding.shape)
    print("First 10 values:")
    print(embedding[:10])

# import cv2

# img = cv2.imread(
#     "dataset/aetid/aetidf_cropped.jpg"
# )

# print(img.shape)