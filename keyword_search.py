import json
import re

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

def keyword_search(query, top_k=5):
    query_tokens = tokenize(query)
    expanded_tokens = expand_query(query_tokens)

    results = []
    for url, caption in captions.items():
        caption_tokens = tokenize(caption)
        matched = [word for word in expanded_tokens if word in caption_tokens]
        score = len(matched)
        if score > 0:
            results.append((url, score))

    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_k]
