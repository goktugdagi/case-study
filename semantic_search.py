import os
import json
from dotenv import load_dotenv
import google.generativeai as genai
import chromadb

load_dotenv()
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

DB_PATH = "chromadb_data"
COLLECTION_NAME = "hotel_image_captions_v2"

chroma_client = chromadb.PersistentClient(path=DB_PATH)
collection = chroma_client.get_collection(name=COLLECTION_NAME)

def get_embedding(text):
    result = genai.embed_content(
        model="models/embedding-001",
        content=text,
        task_type="SEMANTIC_SIMILARITY"
    )
    return result["embedding"]

def semantic_search(query, top_k=5):
    query_embedding = get_embedding(query)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["metadatas", "distances"]
    )
    urls = []
    for i, metadata in enumerate(results["metadatas"][0]):
        urls.append((metadata["url"], 1.0 - results["distances"][0][i]))
    return urls
