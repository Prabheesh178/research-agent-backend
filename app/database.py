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
    
    # Skills Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS user_skills (
        user_id TEXT,
        skill_id TEXT,
        name TEXT NOT NULL,
        trigger_keywords TEXT NOT NULL,
        intent_type TEXT NOT NULL,
        system_prompt_extension TEXT NOT NULL,
        enabled INTEGER DEFAULT 1,
        PRIMARY KEY (user_id, skill_id)
    )
    """)
    
    # Plugins Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS user_plugins (
        user_id TEXT,
        plugin_id TEXT,
        name TEXT NOT NULL,
        tools_provided TEXT NOT NULL,
        auth_type TEXT NOT NULL,
        auth_fields TEXT NOT NULL,
        enabled INTEGER DEFAULT 1,
        auth_data TEXT,
        PRIMARY KEY (user_id, plugin_id)
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
    cursor.execute("DELETE FROM user_skills WHERE user_id = ?", (user_id,))
    cursor.execute("DELETE FROM user_plugins WHERE user_id = ?", (user_id,))
    conn.commit()
    conn.close()

# Skills Helpers
def get_user_skills(user_id: str) -> list:
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM user_skills WHERE user_id = ?", (user_id,))
    rows = cursor.fetchall()
    conn.close()
    
    skills = []
    for r in rows:
        skills.append({
            "skill_id": r["skill_id"],
            "name": r["name"],
            "trigger_keywords": json.loads(r["trigger_keywords"]),
            "intent_type": r["intent_type"],
            "system_prompt_extension": r["system_prompt_extension"],
            "enabled": bool(r["enabled"])
        })
    return skills

def save_user_skill(user_id: str, skill: dict):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT OR REPLACE INTO user_skills 
        (user_id, skill_id, name, trigger_keywords, intent_type, system_prompt_extension, enabled) 
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            user_id,
            skill["id"],
            skill["name"],
            json.dumps(skill["trigger_keywords"]),
            skill["intent_type"],
            skill["system_prompt_extension_content"],
            1 if skill.get("enabled", True) else 0
        )
    )
    conn.commit()
    conn.close()

def delete_user_skill(user_id: str, skill_id: str):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM user_skills WHERE user_id = ? AND skill_id = ?", (user_id, skill_id))
    conn.commit()
    conn.close()

def toggle_skill_enabled(user_id: str, skill_id: str, enabled: bool):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE user_skills SET enabled = ? WHERE user_id = ? AND skill_id = ?",
        (1 if enabled else 0, user_id, skill_id)
    )
    conn.commit()
    conn.close()

# Plugins Helpers
def get_user_plugins(user_id: str) -> list:
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM user_plugins WHERE user_id = ?", (user_id,))
    rows = cursor.fetchall()
    conn.close()
    
    plugins = []
    for r in rows:
        plugins.append({
            "plugin_id": r["plugin_id"],
            "name": r["name"],
            "tools_provided": json.loads(r["tools_provided"]),
            "auth_type": r["auth_type"],
            "auth_fields": json.loads(r["auth_fields"]),
            "enabled": bool(r["enabled"]),
            "auth_data": json.loads(r["auth_data"]) if r["auth_data"] else {}
        })
    return plugins

def save_user_plugin(user_id: str, plugin: dict):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT OR REPLACE INTO user_plugins 
        (user_id, plugin_id, name, tools_provided, auth_type, auth_fields, enabled, auth_data) 
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            user_id,
            plugin["id"],
            plugin["name"],
            json.dumps(plugin["tools_provided"]),
            plugin["auth_type"],
            json.dumps(plugin.get("auth_fields", [])),
            1 if plugin.get("enabled", True) else 0,
            json.dumps(plugin.get("auth_data", {}))
        )
    )
    conn.commit()
    conn.close()

def delete_user_plugin(user_id: str, plugin_id: str):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM user_plugins WHERE user_id = ? AND plugin_id = ?", (user_id, plugin_id))
    conn.commit()
    conn.close()

def toggle_plugin_enabled(user_id: str, plugin_id: str, enabled: bool):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE user_plugins SET enabled = ? WHERE user_id = ? AND plugin_id = ?",
        (1 if enabled else 0, user_id, plugin_id)
    )
    conn.commit()
    conn.close()

def save_plugin_auth(user_id: str, plugin_id: str, auth_data: dict):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE user_plugins SET auth_data = ? WHERE user_id = ? AND plugin_id = ?",
        (json.dumps(auth_data), user_id, plugin_id)
    )
    conn.commit()
    conn.close()

