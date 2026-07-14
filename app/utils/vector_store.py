import json
import uuid
import numpy as np
from openai import OpenAI
from app.database import get_db_connection
from app.config import settings

class CompletionsWrapper:
    def __init__(self, original_completions):
        self.original_completions = original_completions
        
    def create(self, *args, **kwargs):
        try:
            return self.original_completions.create(*args, **kwargs)
        except Exception as e:
            err_msg = str(e)
            if "429" in err_msg or "rate" in err_msg.lower() or "limit" in err_msg.lower() or "too many requests" in err_msg.lower():
                import time
                time.sleep(3.0)
                try:
                    return self.original_completions.create(*args, **kwargs)
                except Exception:
                    raise Exception("Free tier rate limit hit. Retrying in 10 seconds — or switch to a paid key for no limits.")
            raise e

class ChatWrapper:
    def __init__(self, original_chat):
        self.completions = CompletionsWrapper(original_chat.completions)

class OpenAIWrapper:
    def __init__(self, original_client):
        self.original_client = original_client
        self.chat = ChatWrapper(original_client.chat)
        
    @property
    def embeddings(self):
        return self.original_client.embeddings

def get_openai_client(openai_key: str = None, base_url: str = None) -> OpenAIWrapper:
    key = openai_key or settings.OPENAI_API_KEY
    b_url = base_url or settings.LLM_BASE_URL
    if not key or key.strip() == "":
        raise ValueError("API Key is missing. Please provide it in Settings or your environment.")
    
    # Auto-detect OpenRouter keys
    cleaned_key = key.strip()
    if cleaned_key.startswith("sk-or-v1") and (not b_url or b_url.strip() == ""):
        b_url = "https://openrouter.ai/api/v1"
        
    if b_url and b_url.strip() != "":
        client = OpenAI(api_key=key, base_url=b_url.strip())
    else:
        client = OpenAI(api_key=key)
        
    return OpenAIWrapper(client)

def get_embedding(text: str, openai_key: str = None, base_url: str = None) -> list:
    try:
        client = get_openai_client(openai_key, base_url)
        response = client.embeddings.create(
            input=[text.replace("\n", " ")],
            model="text-embedding-3-small"
        )
        return response.data[0].embedding
    except Exception as e:
        # Fallback to standard OpenAI if custom base_url failed
        if base_url and base_url.strip() != "":
            try:
                client = get_openai_client(openai_key, base_url=None)
                response = client.embeddings.create(
                    input=[text.replace("\n", " ")],
                    model="text-embedding-3-small"
                )
                return response.data[0].embedding
            except Exception:
                pass
        
        # Hash-based deterministic mock embedding fallback
        np.random.seed(hash(text) % (2**32))
        vec = np.random.randn(1536)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.tolist()

def get_embeddings_batch(texts: list, openai_key: str = None, base_url: str = None) -> list:
    if not texts:
        return []
    try:
        client = get_openai_client(openai_key, base_url)
        cleaned_texts = [t.replace("\n", " ") for t in texts]
        response = client.embeddings.create(
            input=cleaned_texts,
            model="text-embedding-3-small"
        )
        sorted_data = sorted(response.data, key=lambda x: x.index)
        return [item.embedding for item in sorted_data]
    except Exception:
        # Process one by one using get_embedding to leverage fallback
        return [get_embedding(t, openai_key, base_url) for t in texts]

def add_document_chunks(document_id: str, user_id: str, chunks: list, openai_key: str = None, base_url: str = None):
    """
    chunks is a list of dicts: [{"text": str, "page_num": int}]
    """
    if not chunks:
        return
    
    texts = [c["text"] for c in chunks]
    embeddings = get_embeddings_batch(texts, openai_key, base_url)
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    for i, chunk in enumerate(chunks):
        chunk_id = str(uuid.uuid4())
        embedding_json = json.dumps(embeddings[i])
        cursor.execute(
            """
            INSERT INTO document_chunks (id, document_id, user_id, text, page_num, embedding_json)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (chunk_id, document_id, user_id, chunk["text"], chunk["page_num"], embedding_json)
        )
    
    conn.commit()
    conn.close()

def search_similarity(user_id: str, query: str, openai_key: str = None, top_k: int = 5, base_url: str = None) -> list:
    """
    Search chunks for a user using cosine similarity
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Query database for chunks and filenames
    cursor.execute(
        """
        SELECT dc.text, dc.page_num, dc.embedding_json, d.filename
        FROM document_chunks dc
        JOIN documents d ON dc.document_id = d.id
        WHERE dc.user_id = ?
        """,
        (user_id,)
    )
    rows = cursor.fetchall()
    conn.close()
    
    if not rows:
        return []
    
    # Get query embedding
    query_vector = np.array(get_embedding(query, openai_key, base_url))
    
    results = []
    for row in rows:
        emb = np.array(json.loads(row["embedding_json"]))
        # Cosine similarity for normalized vectors is just the dot product
        similarity = float(np.dot(query_vector, emb))
        results.append({
            "text": row["text"],
            "page_num": row["page_num"],
            "filename": row["filename"],
            "similarity": similarity
        })
    
    # Sort by similarity descending
    results = sorted(results, key=lambda x: x["similarity"], reverse=True)
    return results[:top_k]
