import json
import faiss

INDEX_PATH = "database/faces.faiss"
METADATA_PATH = "database/metadata.json"

index = faiss.read_index(INDEX_PATH)

with open(METADATA_PATH, "r") as f:
    metadata = json.load(f)

print("Loaded FAISS index with", index.ntotal, "vectors")
print("Embedding dimension:", index.d)

for i in range(index.ntotal):
    embedding = index.reconstruct(i)
    print(
        f"Embedding #{i}: shape={embedding.shape}, "
        f"metadata={metadata[i] if i < len(metadata) else 'missing'}"
    )