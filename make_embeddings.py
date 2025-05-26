#!/usr/bin/env python3
import os
import pickle
from dotenv import load_dotenv
import openai

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# 1) Load your chunks
with open("metadata.pkl", "rb") as f:
    metadata = pickle.load(f)

print(f"Loaded metadata with {len(metadata)} chunks")  # sanity check

# 2) Embed each chunk
embeddings = []
for idx, item in enumerate(metadata, start=1):
    # create embedding
    resp = openai.embeddings.create(
        model="text-embedding-3-small",
        input=item["text"]
    )
    # append the vector
    embeddings.append(resp.data[0].embedding)
    if idx % 20 == 0 or idx == len(metadata):
        print(f"  → processed {idx}/{len(metadata)}")   # progress indicator

# 3) Save embeddings
with open("embeddings.pkl", "wb") as f:
    pickle.dump(embeddings, f)

print(f"Saved {len(embeddings)} embeddings to embeddings.pkl")

