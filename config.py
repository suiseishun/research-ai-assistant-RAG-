import os
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("API Key not found. Please check your .env file.")

EMBEDDING_MODEL = "models/text-embedding-004" 
GENERATION_MODEL = "gemini-flash-latest"
DIMENSION = 3072  

DATA_DIR = "./data"
PDF_SOURCE_DIR = os.path.join(DATA_DIR, "raw_pdfs")
DB_DIR = os.path.join(DATA_DIR, "vector_db")
COLLECTION_NAME = "research_papers_db"

os.makedirs(PDF_SOURCE_DIR, exist_ok=True)
os.makedirs(DB_DIR, exist_ok=True)