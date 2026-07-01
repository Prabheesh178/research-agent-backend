import os
import sqlite3
import json

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "memory.db")

DEFAULT_PROFILE = {
    "vocab_level": "postgraduate",
    "citation_style": "IEEE",
    "avg_sentence_length": 20,
    "connectors": ["however", "furthermore", "notably", "this suggests"],
    "session_count": 0,
    "domain": "general academic",
    "writing_style": "formal",
    "sentence_variety": "moderate",
    "writing_quirks": "",
    "topics_researched": []
}

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # User Profiles table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS user_profiles (
        user_id TEXT PRIMARY KEY,
        profile_json TEXT NOT NULL
    )
    """)
    
    # Documents metadata table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS documents (
        id TEXT PRIMARY KEY,
        user_id TEXT NOT NULL,
        filename TEXT NOT NULL,
        upload_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)
    
    # Document chunks for vector storage
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS document_chunks (
        id TEXT PRIMARY KEY,
        document_id TEXT NOT NULL,
        user_id TEXT NOT NULL,
        text TEXT NOT NULL,
        page_num INTEGER NOT NULL,
        embedding_json TEXT NOT NULL,
        FOREIGN KEY (document_id) REFERENCES documents (id) ON DELETE CASCADE
    )
    """)
    
    conn.commit()
    conn.close()

# Initialize database on import
init_db()

# DB Helper Functions
def get_user_profile(user_id: str) -> dict:
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT profile_json FROM user_profiles WHERE user_id = ?", (user_id,))
    row = cursor.fetchone()
    conn.close()
    
    if row:
        return json.loads(row["profile_json"])
    return None

def save_user_profile(user_id: str, profile: dict):
    conn = get_db_connection()
    cursor = conn.cursor()
    profile_json = json.dumps(profile)
    cursor.execute(
        "INSERT OR REPLACE INTO user_profiles (user_id, profile_json) VALUES (?, ?)",
        (user_id, profile_json)
    )
    conn.commit()
    conn.close()

def delete_user_data(user_id: str):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM document_chunks WHERE user_id = ?", (user_id,))
    cursor.execute("DELETE FROM documents WHERE user_id = ?", (user_id,))
    cursor.execute("DELETE FROM user_profiles WHERE user_id = ?", (user_id,))
    conn.commit()
    conn.close()
