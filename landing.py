from flask import Flask, render_template, redirect, send_from_directory
import subprocess
import threading
import time
import os
import logging
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

app = Flask(__name__, static_folder='static')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add your API key
GOOGLE_API_KEY = "AIzaSyDCRZOok-jCCpl-q5kCSXR9fhA3lpdflvY"

# Configure Google Generative AI
genai.configure(api_key=GOOGLE_API_KEY)

def initialize_vector_store():
    """Initialize and test vector store"""
    try:
        embeddings = GoogleGenerativeAIEmbeddings(
            google_api_key=GOOGLE_API_KEY,
            model="models/embedding-001"
        )
        
        try:
            vectorstore = FAISS.load_local(
                folder_path="Faiss",
                embeddings=embeddings
            )
            logger.info("Vector store loaded successfully")
            
            # Test search functionality
            try:
                test_query = "test query"
                results = vectorstore.similarity_search(test_query, k=1)
                logger.info(f"Search test successful, found {len(results)} results")
                return vectorstore
            except Exception as search_error:
                logger.error(f"Search error: {str(search_error)}")
                return None
                
        except Exception as e:
            logger.error(f"Error loading vector store: {str(e)}")
            return None
            
    except Exception as e:
        logger.error(f"Error initializing embeddings: {str(e)}")
        return None

def run_streamlit():
    """Run Streamlit server"""
    try:
        subprocess.run(["streamlit", "run", "Vakeel.py"])
    except Exception as e:
        logger.error(f"Error starting Streamlit: {e}")

@app.route('/')
def home():
    return render_template('landing_page.html')

@app.route('/app')
def launch_app():
    return redirect('http://localhost:8501')

@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory('static', filename)

if __name__ == '__main__':
    # Create static folder if it doesn't exist
    if not os.path.exists('static'):
        os.makedirs('static')
    
    # Initialize vector store
    vectorstore = initialize_vector_store()
    if vectorstore is None:
        logger.warning("Vector store initialization failed, some features may not work")
    
    # Start Streamlit in a separate thread
    streamlit_thread = threading.Thread(target=run_streamlit)
    streamlit_thread.daemon = True
    streamlit_thread.start()
    
    # Wait for Streamlit to start
    time.sleep(3)
    
    # Run Flask app
    app.run(port=5000) 