import json
import re
import os
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

with open("hotel_image_captions_output.json", "r", encoding="utf-8") as f:
    captions = json.load(f)

synonyms = {
    "çift": ["double", "çift", "2 kişilik", "iki kişilik"],
    "tek": ["single", "tek"],
    "üç": ["triple", "üç", "3 kişilik"],
    "balkon": ["balcony"],
    "klima": ["air conditioning", "ac"],
    "deniz": ["sea", "ocean"],
    "şehir": ["city", "urban"],
    "doğa": ["nature", "forest"],
    "masa": ["desk", "table"],
    "yatak": ["bed"],
    "tv": ["televizyon", "tv"]
}

def tokenize(text):
    return re.findall(r'\b\w+\b', text.lower())

def expand_query(tokens):
    expanded = set(tokens)
    for token in tokens:
        for base, syns in synonyms.items():
            if token in syns or token == base:
                expanded.update(syns)
                expanded.add(base)
    return list(expanded)

def get_embedding(text):
    result = genai.embed_content(
        model="models/embedding-001",
        content=text,
        task_type="SEMANTIC_SIMILARITY"
    )
    return result["embedding"]

def hybrid_search(user_query, top_k=5, weight_keyword=0.5, weight_semantic=0.5):
    query_tokens = tokenize(user_query.lower())
    expanded_tokens = expand_query(query_tokens)

    keyword_scores = {}
    for url, caption in captions.items():
        caption_tokens = tokenize(caption)
        matched = [word for word in expanded_tokens if word in caption_tokens]
        score = len(matched)
        if score > 0:
            keyword_scores[url] = score

    max_keyword_score = max(keyword_scores.values(), default=1)

    sem_scores = {}
    query_embedding = get_embedding(user_query)
    sem_results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k * 2,
        include=["metadatas", "distances"]
    )

    max_distance = max(sem_results["distances"][0]) if sem_results["distances"][0] else 1.0
    for i, metadata in enumerate(sem_results["metadatas"][0]):
        url = metadata["url"]
        distance = sem_results["distances"][0][i]
        sim_score = 1.0 - (distance / max_distance)
        sem_scores[url] = sim_score

    hybrid_scores = {}
    all_urls = set(keyword_scores.keys()).union(set(sem_scores.keys()))
    for url in all_urls:
        k_score = keyword_scores.get(url, 0) / max_keyword_score
        s_score = sem_scores.get(url, 0)
        total_score = (weight_keyword * k_score) + (weight_semantic * s_score)
        hybrid_scores[url] = total_score

    sorted_results = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return sorted_results
