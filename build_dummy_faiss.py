import faiss
import numpy as np

# Create fake embeddings (3 docs x 768 dims)
dim = 768
embeddings = np.random.random((3, dim)).astype('float32')

# Build index
index = faiss.IndexFlatL2(dim)
index.add(embeddings)

# Save
faiss.write_index(index, "data/wikipedia.contriever")
print("✅ Dummy FAISS index saved.")
