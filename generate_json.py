import json

# Vertex AI Vector Search requires an initial JSONL file with 768 dimensions
datapoint = {
    "id": "init_0001",
    "embedding": [0.0] * 768
}

with open("init_embeddings.json", "w") as f:
    f.write(json.dumps(datapoint) + "\n")

print("Generated init_embeddings.json with 768-dimensional dummy vector.")
