#!/usr/bin/env python3
import os
import pickle
import logging
from dotenv import load_dotenv
import openai

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    logger.error("OPENAI_API_KEY not set")
    raise RuntimeError("OPENAI_API_KEY not set")

try:
    # Define disk path for Render
    disk_path = "/app/data"  # Adjust if your disk mount path differs
    metadata_path = f"{disk_path}/metadata.pkl"
    embeddings_path = f"{disk_path}/embeddings.pkl"

    # 1) Load metadata
    logger.info(f"Loading metadata from {metadata_path}")
    with open(metadata_path, "rb") as f:
        metadata = pickle.load(f)
    logger.info(f"Loaded metadata with {len(metadata)} chunks")

    # 2) Embed each chunk
    embeddings = []
    for idx, item in enumerate(metadata, start=1):
        try:
            logger.info(f"Processing chunk {idx}/{len(metadata)}")
            resp = openai.embeddings.create(
                model="text-embedding-3-small",
                input=item["text"]
            )
            embeddings.append(resp.data[0].embedding)
            if idx % 20 == 0 or idx == len(metadata):
                logger.info(f"  → processed {idx}/{len(metadata)}")
        except Exception as e:
            logger.error(f"Error processing chunk {idx}: {str(e)}")
            raise

    # 3) Save embeddings
    logger.info(f"Saving {len(embeddings)} embeddings to {embeddings_path}")
    with open(embeddings_path, "wb") as f:
        pickle.dump(embeddings, f)
    logger.info("Embeddings saved successfully")

except Exception as e:
    logger.error(f"Script failed: {str(e)}")
    raise
