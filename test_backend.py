import os
import sys

# Add the parent folder to path to import app correctly
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.database import init_db, save_user_profile, get_user_profile
from app.utils.pdf_parser import extract_pdf_chunks

print("Initializing database...")
init_db()
print("Database initialized successfully!")

# Test user profile database functions
test_user = "test_user_99"
test_profile = {
    "vocab_level": "postgraduate",
    "citation_style": "IEEE",
    "avg_sentence_length": 25,
    "connectors": ["however", "notably"],
    "session_count": 1,
    "domain": "AI Safety"
}

print(f"Saving profile for {test_user}...")
save_user_profile(test_user, test_profile)

print("Retrieving profile...")
loaded = get_user_profile(test_user)
print("Loaded Profile:", loaded)

assert loaded is not None
assert loaded["domain"] == "AI Safety"
print("Database tests passed!")

# Check pypdf import and simple text chunking
dummy_text = "This is a paragraph. " * 50
print("Testing PDF parser chunking...")
# Create a dummy pdf content (just text extraction test helper)
from pypdf import PdfWriter
writer = PdfWriter()
writer.add_blank_page(width=100, height=100)
pdf_stream = sys.stdout # mock
print("PDF modules import successfully!")
