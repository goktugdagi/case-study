import os
from dotenv import load_dotenv
import google.generativeai as genai
from PIL import Image
from io import BytesIO
import requests
import json
import chromadb
import time 


load_dotenv()


GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in .env file or environment variables. Please ensure it is set.")


image_urls = [
    "http://static.obilet.com.s3.eu-central-1.amazonaws.com/CaseStudy/HotelImages/1.jpg",
    "http://static.obilet.com.s3.eu-central-1.amazonaws.com/CaseStudy/HotelImages/25.jpg"
]


try:
    genai.configure(api_key=GEMINI_API_KEY)

    caption_model = genai.GenerativeModel("gemini-1.5-flash") 
except Exception as e:
    print(f"Error configuring Gemini or initializing models: {e}")
    exit() 


try:
    
    chroma_client = chromadb.PersistentClient(path="chromadb_data")
    
    collection = chroma_client.get_or_create_collection(
        name="hotel_image_captions_v2", 
        
    )
except Exception as e:
    print(f"Error initializing ChromaDB: {e}")
    exit() 


def generate_caption_from_url(image_url: str, retries=5, delay=5):
    """
    Downloads an image from a URL and generates a detailed caption using Gemini.
    Includes basic retry logic for network issues.
    """
    for attempt in range(retries):
        try:

            response = requests.get(image_url, timeout=20) 
            response.raise_for_status()  
            

            image = Image.open(BytesIO(response.content))


            prompt = """
                    Lütfen aşağıdaki otel odasını hem Türkçe hem İngilizce olarak detaylı açıkla. Aşağıdaki formatı kullan:
                    - Bu otel odasını detaylı bir şekilde açıkla. Özelliklerini, manzarasını, yatak tipini ve varsa balkon/masa/klima gibi şeyleri belirt.
            """

            generation_config = genai.types.GenerationConfig(
                temperature=0.4, 
            )
            safety_settings = [ 
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            ]

            response = caption_model.generate_content(
                [prompt, image],
                generation_config=generation_config,
                safety_settings=safety_settings, 
                stream=False 
            )
            
     
            if response.parts:
                return response.text.strip()
            else:
                
                print(f"No text part in response for {image_url}. Finish reason: {response.prompt_feedback}")
                return "Açıklama üretilemedi." 

        except requests.exceptions.RequestException as e:
            print(f"Attempt {attempt + 1} failed for {image_url}: {e}")
            if attempt < retries - 1:
                time.sleep(delay) 
            else:
                print(f"Failed to download image after {retries} attempts: {image_url}")
                raise 
        except Exception as e: 
            print(f"Error generating caption for {image_url}: {e}")
            raise 

def get_embedding(text: str):
    """
    Generates an embedding for the given text using Gemini's embedding model.
    """
    try:
        
        result = genai.embed_content(
            model="models/embedding-001",  
            content=text,
            task_type="SEMANTIC_SIMILARITY" 

        )
        return result["embedding"]
    except Exception as e:
        print(f"Error generating embedding for text '{text[:50]}...': {e}")
        raise 


captions_dict = {} 

print("🚀 Starting image processing...")

for idx, url in enumerate(image_urls):
    print(f"({idx + 1}/{len(image_urls)}) Processing: {url}")
    try:
    
        caption = generate_caption_from_url(url)
        if not caption or caption == "Açıklama üretilemedi.":
            print(f"Skipping embedding for {url} due to missing caption.")
            captions_dict[url] = caption 
            continue


        embedding = get_embedding(caption)

    
        collection.upsert(
            ids=[f"image_{idx}"],       
            documents=[caption],       
            metadatas=[{"url": url, "source": "hotel_images"}], 
            embeddings=[embedding]      
        )
        captions_dict[url] = caption
        print(f"SUCCESS: {url} => \"{caption[:100]}...\"") 

    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP Error for {url}: {http_err}")
        captions_dict[url] = f"HTTP Error: {http_err}"
    except requests.exceptions.ConnectionError as conn_err:
        print(f"Connection Error for {url}: {conn_err}")
        captions_dict[url] = f"Connection Error: {conn_err}"
    except requests.exceptions.Timeout as timeout_err:
        print(f"Timeout Error for {url}: {timeout_err}")
        captions_dict[url] = f"Timeout Error: {timeout_err}"
    except Exception as e:
        
        print(f"An unexpected error occurred for {url}: {e}")
        captions_dict[url] = f"Error: {e}" 


output_filename = "hotel_image_captions_output.json"
try:
    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(captions_dict, f, ensure_ascii=False, indent=4)
    print(f"Captions and errors saved to {output_filename}")
except IOError as e:
    print(f"Error saving captions to JSON: {e}")

print("Processing complete.")

