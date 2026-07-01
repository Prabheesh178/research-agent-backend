import io
import re
from pypdf import PdfReader

def extract_pdf_chunks(file_bytes: bytes, filename: str) -> list:
    """
    Parse a PDF from bytes and return list of chunks: [{"text": str, "page_num": int}]
    """
    reader = PdfReader(io.BytesIO(file_bytes))
    chunks = []
    
    for page_idx, page in enumerate(reader.pages):
        page_num = page_idx + 1
        text = page.extract_text()
        if not text:
            continue
        
        # Clean text
        text = re.sub(r'\s+', ' ', text).strip()
        if len(text) < 50:
            continue
            
        # Split page text into chunks of roughly 400 words
        words = text.split(' ')
        chunk_size = 400
        for i in range(0, len(words), chunk_size):
            chunk_words = words[i:i + chunk_size]
            chunk_text = ' '.join(chunk_words).strip()
            if len(chunk_text) > 30:
                chunks.append({
                    "text": chunk_text,
                    "page_num": page_num
                })
                
    return chunks
