import os
import logging
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

GOOGLE_API_KEY = "AIzaSyDCRZOok-jCCpl-q5kCSXR9fhA3lpdflvY"

def create_dummy_database():
    """Create a minimal working database"""
    try:
        # Ensure Faiss directory exists
        os.makedirs("Faiss", exist_ok=True)
        
        # Create embeddings
        embeddings = GoogleGenerativeAIEmbeddings(
            google_api_key=GOOGLE_API_KEY,
            model="models/embedding-001"
        )
        
        # Create sample documents with more legal content
        texts = [
            "The right to freedom of speech is protected under Article 19 of the Indian Constitution.",
            "Habeas corpus is a legal remedy to protect against unlawful detention.",
            "The right to equality is enshrined in Article 14 of the Indian Constitution.",
            "The principle of natural justice requires fair hearing and unbiased decision making."
        ]
        
        metadata = [
            {"type": "legal_principle", "source": "Constitution", "article": "19"},
            {"type": "legal_principle", "source": "Common Law", "remedy": "habeas_corpus"},
            {"type": "legal_principle", "source": "Constitution", "article": "14"},
            {"type": "legal_principle", "source": "Common Law", "principle": "natural_justice"}
        ]
        
        # Create FAISS index
        vectorstore = FAISS.from_texts(
            texts=texts,
            embedding=embeddings,
            metadatas=metadata
        )
        
        # Save the vector store
        vectorstore.save_local("Faiss")
        logger.info("Successfully created dummy database")
        return True
        
    except Exception as e:
        logger.error(f"Error creating dummy database: {e}")
        return False

if __name__ == "__main__":
    success = create_dummy_database()
    if success:
        logger.info("Database created successfully")
    else:
        logger.error("Failed to create database") 