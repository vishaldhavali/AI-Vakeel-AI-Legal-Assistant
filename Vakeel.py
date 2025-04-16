import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import os
import hashlib
import time
from time import perf_counter
import json
from db import (
    create_db, 
    add_user, 
    get_user, 
    add_chat, 
    get_chat_history, 
    clear_chat_history, 
    update_password,
    get_total_messages,
    get_account_age,
    update_profile_image,
    update_user_email_with_verification,
    delete_user_account,
    list_sessions,
    verify_profile_image,
    get_user_email,
)
import base64
import streamlit_lottie
from streamlit_lottie import st_lottie
import requests
from streamlit_option_menu import option_menu
import sqlite3
import logging
import re
from functools import wraps
from utils.session_manager import SessionManager
from datetime import datetime
from langchain.chains import LLMChain
import warnings

import faiss as faiss

import numpy as np
from data.dummy_data import (
    get_court_vacancy,
    get_case_status,
    get_traffic_violation,
    get_fast_track_courts,
    get_live_streams,
    get_random_case_number,
    get_random_violation_id,
    get_legal_aid_centers,  # New function
    get_prison_statistics,  # New function
    match_state_from_query,
)
from typing import Dict, List, Any, Optional
import random

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger("faiss").setLevel(logging.ERROR)

# Update page config with new branding
st.set_page_config(page_title="AI-Vakeel", page_icon=":scales:", layout="wide")

# At the top of the file, keep the direct API key configuration
GOOGLE_API_KEY = "AIzaSyDCRZOok-jCCpl-q5kCSXR9fhA3lpdflvY"

# Configure API key directly
genai.configure(api_key=GOOGLE_API_KEY)

create_db()

# Add a proper logging configuration
logging.basicConfig(
    filename="ai_vakeel.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# Initialize session manager
session_manager = SessionManager()

# Initialize embeddings and vector store at module level
try:
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=GOOGLE_API_KEY
    )
    vectorstore = FAISS.load_local("Faiss", embeddings)
    logging.info("Vector store initialized successfully")
except Exception as e:
    logging.error(f"Error initializing vector store: {str(e)}")
    vectorstore = None

# Document and DocStore classes for compatibility with the pickle file
class Document:
    """Simple document class to match LangChain's format"""

    def __init__(self, page_content: str, metadata: Dict[str, Any] = None):
        self.page_content = page_content
        self.metadata = metadata or {}


class DocStore:
    """Simple document store to match LangChain's format"""

    def __init__(self):
        self._dict = {}
        
    def add(self, doc_id: str, doc: Document):
        self._dict[doc_id] = doc
        
    def get(self, doc_id: str) -> Optional[Document]:
        return self._dict.get(doc_id)
        
    def __getstate__(self):
        return {"_dict": self._dict}
        
    def __setstate__(self, state):
        self._dict = state["_dict"]


def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()


# Update the get_conversational_chain function
def get_conversational_chain():
    try:
        model = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=0.7,
            google_api_key=GOOGLE_API_KEY,
            generation_config={"max_output_tokens": 2048, "top_p": 0.85, "top_k": 40},
        )
        return model
    except Exception as e:
        logging.error(f"Error creating conversation chain: {e}")
        return None


def rate_limit(limit_seconds=1):
    last_request_time = {}
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            current_time = time.time()
            if func.__name__ in last_request_time:
                time_passed = current_time - last_request_time[func.__name__]
                if time_passed < limit_seconds:
                    time.sleep(limit_seconds - time_passed)
            
            last_request_time[func.__name__] = current_time
            return func(*args, **kwargs)

        return wrapper

    return decorator


def monitor_performance(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = perf_counter()
        result = func(*args, **kwargs)
        end_time = perf_counter()
        
        logging.info(
            f"Function {func.__name__} took {end_time - start_time:.2f} seconds to execute"
        )
        return result

    return wrapper


@rate_limit(1)
@monitor_performance
def user_input(user_question):
    try:
        # First check if this is a query that should use dummy data
        doj_response = handle_doj_query(user_question)
        if doj_response:
            # If we have a DoJ-specific response with dummy data, return it
            logging.info(f"Using dummy data for query: {user_question}")
            return doj_response

        # Continue with normal processing for other queries
        # Get context from vector store
        context = enhance_context_retrieval(user_question)
        
        # Get enhanced response using both trained data and internet context
        response = get_enhanced_response(user_question, context)
        
        # Log the interaction
        logging.info(f"Query: {user_question}")
        logging.info(f"Context Length: {len(context) if context else 0}")
        
        return response
        
    except Exception as e:
        logging.error(f"Error in user_input: {str(e)}")
        return "I apologize, but I'm having trouble processing your request. Please try again."


def display_typing_effect(response):
    output = ""
    response_container = st.empty()
    for char in response:
        output += char
        response_container.markdown(output, unsafe_allow_html=True)
        time.sleep(0.005)
    return output


def sanitized_filename(filename):
    """Sanitize filename removing invalid characters"""
    # Remove invalid characters
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, "_")
    # Remove control characters
    filename = "".join(char for char in filename if ord(char) >= 32)
    return filename


def save_session(username, session_name, session_data):
    """Save chat session with proper validation and error handling"""
    try:
        if not all([username, session_data]):
            logging.warning("Missing required data for saving session")
            return False
        
        # Generate session name from first user message if not provided
        if not session_name or session_name == "New Chat":
            for message in session_data:
                if message.get("role") == "user":
                    session_name = generate_session_name(message.get("content", ""))
                    break
        
        # Create sessions directory if it doesn't exist
        os.makedirs("sessions", exist_ok=True)
        
        # Create a safe filename with timestamp
        safe_username = sanitized_filename(username.replace("@", "_at_"))
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{safe_username}_{sanitized_filename(session_name)}_{timestamp}.json"
        
        # Prepare session info with metadata
        session_info = {
            "username": username,
            "session_name": session_name,
            "timestamp": datetime.now().isoformat(),
            "messages": session_data,
            "metadata": {
                "message_count": len(session_data),
                "last_updated": datetime.now().isoformat()
            }
        }
        
        # Save to file with proper encoding
        filepath = os.path.join("sessions", filename)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(session_info, f, indent=2, ensure_ascii=False)
        
        logging.info(f"Session saved successfully: {filename}")
        return True
        
    except Exception as e:
        logging.error(f"Error saving session: {str(e)}")
        return False


def load_session(session_id):
    """Load chat session from file"""
    try:
        file_path = os.path.join("sessions", session_id)
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return None
    except Exception as e:
        logging.error(f"Error loading session: {str(e)}")
        return None


def delete_session(session_id):
    """Delete a chat session"""
    try:
        file_path = os.path.join("sessions", session_id)
        if os.path.exists(file_path):
            os.remove(file_path)
            logging.info(f"Session deleted: {session_id}")
            return True
        return False
    except Exception as e:
        logging.error(f"Error deleting session: {str(e)}")
        return False


def delete_all_sessions(username):
    """Delete all sessions for a user"""
    try:
        sanitized_username = sanitized_filename(username)
        sessions_dir = "sessions"
        for filename in os.listdir(sessions_dir):
            if filename.startswith(f"{sanitized_username}_"):
                file_path = os.path.join(sessions_dir, filename)
                os.remove(file_path)
        logging.info(f"All sessions deleted for user: {username}")
        return True
    except Exception as e:
        logging.error(f"Error deleting all sessions: {str(e)}")
        return False


def export_chat_history(username, format="json"):
    if format == "json":
        history = get_chat_history(username)
        filename = f"chat_history_{username}_{int(time.time())}.json"
        with open(filename, "w") as f:
            json.dump(history, f, indent=2)
        return filename
    # Add support for other formats (CSV, PDF, etc.)


st.markdown(
    """
<style>
.main {
    background-color: #f5f7ff;
}
.chat-message-user {
    background-color: #007bff;
    color: white;
    border-radius: 15px;
    padding: 15px;
    margin: 10px 0;
    box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
}
.chat-message-assistant {
    background-color: #ffffff;
    border-radius: 15px;
    padding: 15px;
    margin: 10px 0;
    box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    border-left: 5px solid #007bff;
}
.stButton>button {
    background-color: #007bff;
    color: white;
    border-radius: 25px;
    padding: 10px 25px;
    transition: all 0.3s ease;
}
.stButton>button:hover {
    transform: translateY(-2px);
    box-shadow: 0 5px 15px rgba(0,0,0,0.2);
}
/* Fix navigation highlighting */
.stNavigationMenu {
    background-color: transparent !important;
}
.stNavigationMenu button[data-baseweb="tab"] {
    background-color: transparent !important;
}
.stNavigationMenu button[data-baseweb="tab"][aria-selected="true"] {
    background-color: #007bff !important;
    color: white !important;
}

/* Profile section improvements */
.profile-sidebar {
    background-color: transparent;
    border-radius: 10px;
    padding: 20px;
}
.profile-image-container {
    text-align: center;
    margin-bottom: 20px;
}
.profile-image-container img {
    width: 150px;
    height: 150px;
    border-radius: 75px;
    object-fit: cover;
    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
}
.profile-name {
    text-align: center;
    font-size: 1.2em;
    font-weight: 600;
    color: #2c3e50;
    margin: 10px 0;
}
.upload-section {
    margin-top: 20px;
    padding: 15px;
    border-radius: 8px;
    background-color: #f8f9fa;
}

/* Professional Sidebar Styling */
.sidebar-container {
    background: linear-gradient(180deg, #1a237e 0%, #0d47a1 100%);
    border-radius: 15px;
    padding: 20px;
    color: white;
    margin: -1rem -1rem 1rem -1rem;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    position: relative;
    padding-top: 80px;
}

.profile-header {
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 20px 0;
    border-bottom: 1px solid rgba(255, 255, 255, 0.1);
    margin-bottom: 20px;
    position: relative;
}

.profile-image-wrapper {
    width: 120px;
    height: 120px;
    position: absolute;
    top: -60px;
    left: 50%;
    transform: translateX(-50%);
    margin-bottom: 15px;
    z-index: 1;
}

.profile-image {
    width: 100%;
    height: 100%;
    border-radius: 50%;
    overflow: hidden;
    border: 4px solid white;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    background-color: white;
}

.profile-image img {
    width: 100%;
    height: 100%;
    object-fit: cover;
}

.username {
    font-size: 1.2rem;
    font-weight: 600;
    color: white;
    text-align: center;
    margin-top: 10px;
}

.chat-session {
    background: rgba(255, 255, 255, 0.1);
    padding: 12px;
    border-radius: 8px;
    margin-bottom: 8px;
    cursor: pointer;
    transition: all 0.3s ease;
}

.chat-session:hover {
    background: rgba(255, 255, 255, 0.2);
    transform: translateX(5px);
}

/* Sidebar chat session styling */
.chat-session-container {
    background: rgba(41, 98, 255, 0.1);
    border-radius: 10px;
    padding: 10px;
    margin-bottom: 10px;
    transition: all 0.3s ease;
}

.chat-session-container:hover {
    background: rgba(41, 98, 255, 0.2);
    transform: translateX(5px);
}

.chat-session-button {
    background: transparent !important;
    border: none !important;
    color: #E0E0E0 !important;
    text-align: left !important;
    padding: 0 !important;
}

.chat-session-button:hover {
    color: #FFFFFF !important;
}

.delete-button {
    background: rgba(255, 59, 48, 0.1) !important;
    border-radius: 5px !important;
    color: #FF3B30 !important;
    transition: all 0.3s ease;
}

.delete-button:hover {
    background: rgba(255, 59, 48, 0.2) !important;
    transform: scale(1.05);
}

/* Session info styling */
.session-info {
    background: linear-gradient(135deg, rgba(41, 98, 255, 0.1) 0%, rgba(13, 71, 161, 0.1) 100%);
    border-radius: 8px;
    padding: 8px 12px;
    margin-bottom: 15px;
    font-size: 0.9em;
    color: #E0E0E0;
}

/* No chats message styling */
.no-chats-message {
    text-align: center;
    padding: 20px;
    color: #666;
    font-style: italic;
}
</style>
""",
    unsafe_allow_html=True,
)


# Update the THINKING_ANIMATION_URL constant
THINKING_ANIMATION_URL = "https://assets5.lottiefiles.com/packages/lf20_kq5rGs.json"

def load_lottie_url(url):
    """Load Lottie animation from URL with better error handling"""
    try:
        r = requests.get(url, timeout=5)  # Add timeout
        if r.status_code == 200:
            animation = r.json()
            if isinstance(animation, dict) and "v" in animation:  # Basic Lottie validation
                return animation
            logging.warning("Invalid Lottie JSON format")
            return None
    except Exception as e:
        logging.error(f"Error loading Lottie animation: {e}")
    return None


# Add these new functions for profile management
def update_user_email(username, new_email, password):
    hashed_password = hash_password(password)
    return update_user_email_with_verification(username, new_email, hashed_password)


def update_user_password(username, old_password, new_password):
    conn = sqlite3.connect("app.db")
    c = conn.cursor()
    try:
        c.execute("SELECT password FROM users WHERE username=?", (username,))
        stored_password = c.fetchone()
        if stored_password and stored_password[0] == hash_password(old_password):
            c.execute(
                "UPDATE users SET password=? WHERE username=?",
                (hash_password(new_password), username),
            )
            conn.commit()
            return True
        return False
    finally:
        conn.close()


def delete_user_account(username, password):
    conn = sqlite3.connect("app.db")
    c = conn.cursor()
    try:
        c.execute("SELECT password FROM users WHERE username=?", (username,))
        stored_password = c.fetchone()
        if stored_password and stored_password[0] == hash_password(password):
            c.execute("DELETE FROM users WHERE username=?", (username,))
            c.execute("DELETE FROM chat_history WHERE username=?", (username,))
            conn.commit()
            return True
        return False
    finally:
        conn.close()


def update_profile_image(username, image_data):
    conn = sqlite3.connect("app.db")
    c = conn.cursor()
    try:
        c.execute(
            "UPDATE users SET profile_image=? WHERE username=?", (image_data, username)
        )
        conn.commit()
        return True
    except sqlite3.Error:
        return False
    finally:
        conn.close()


def allowed_file(filename):
    ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def add_feedback_system():
    st.markdown("### Was this response helpful?")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("👍 Yes"):
            # Store positive feedback
            store_feedback(st.session_state.username, "positive")
            st.success("Thank you for your feedback!")
    with col2:
        if st.button("👎 No"):
            # Store negative feedback
            feedback = st.text_area("Please tell us how we can improve:")
            if st.button("Submit"):
                store_feedback(st.session_state.username, "negative", feedback)
                st.success("Thank you for your feedback!")


def get_legal_citations(response):
    """Extract and format legal citations from the response"""
    # Add regex patterns to identify legal citations
    citation_pattern = r"\b\d{4}\s+\(\d+\)\s+SCC\s+\d+\b"  # Example pattern for Supreme Court citations
    citations = re.findall(citation_pattern, response)
    return citations


def format_response_with_citations(response):
    citations = get_legal_citations(response)
    if citations:
        response += "\n\nCitations:\n" + "\n".join(citations)
    return response


def track_session_analytics(username, session_data):
    """Track user session analytics"""
    analytics = {
        "session_duration": time.time() - session_data.get("start_time", time.time()),
        "message_count": len(session_data.get("messages", [])),
        "query_types": analyze_query_types(session_data.get("messages", [])),
        "timestamp": time.time(),
    }
    store_analytics(username, analytics)


def handle_api_error(error):
    error_messages = {
        "API_KEY_INVALID": "There was an authentication error. Please contact support.",
        "QUOTA_EXCEEDED": "We've reached our query limit. Please try again later.",
        "INVALID_ARGUMENT": "There was an error processing your request. Please try rephrasing.",
        "DEFAULT": "An unexpected error occurred. Please try again later.",
    }
    error_type = str(error).split(":")[0]
    return error_messages.get(error_type, error_messages["DEFAULT"])


def enhance_context_retrieval(query: str) -> str:
    """Enhanced context retrieval with specialized handling for different query types"""
    try:
        logging.info(f"Starting enhanced context retrieval for query: {query}")
        
        # Check for specific query types
        article_match = re.search(r'article\s+(\d+)', query.lower())
        section_match = re.search(r'section\s+(\d+)', query.lower())
        case_law_search = "case" in query.lower() or "judgment" in query.lower() or "precedent" in query.lower()
        
        # Get embedding function
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=GOOGLE_API_KEY
        )
        
        # Load vector store without allow_dangerous_deserialization
        vectorstore = FAISS.load_local(
            "Faiss",
            embeddings
        )
        logging.info("Vector store loaded successfully")

        # Collection to store all results
        combined_results = []
        
        # Special handling for different query types
        if article_match:
            article_number = article_match.group(1)
            logging.info(f"Article-specific query detected: Article {article_number}")
            
            # 1. Get constitution articles
            constitution_query = f"article {article_number} constitution of india"
            try:
                constitution_results = vectorstore.similarity_search(
                    constitution_query,
                    k=2
                )
                combined_results.extend(constitution_results)
                logging.info(f"Retrieved {len(constitution_results)} constitution documents")
            except Exception as e:
                logging.error(f"Error retrieving constitution documents: {str(e)}")
            
            # 2. Get case laws specific to this article
            case_law_query = f"landmark judgments supreme court article {article_number} constitution"
            try:
                case_results = vectorstore.similarity_search(
                    case_law_query,
                    k=3,
                    filter={"type": "case_study"}
                )
                combined_results.extend(case_results)
                logging.info(f"Retrieved {len(case_results)} case studies for Article {article_number}")
            except Exception as e:
                logging.error(f"Error retrieving case studies: {str(e)}")
                
        elif section_match:
            section_number = section_match.group(1)
            logging.info(f"Section-specific query detected: Section {section_number}")
            
            # 1. Get legal documents about this section
            section_query = f"section {section_number} IPC CrPC indian law"
            try:
                section_results = vectorstore.similarity_search(
                    section_query,
                    k=2
                )
                combined_results.extend(section_results)
                logging.info(f"Retrieved {len(section_results)} documents about Section {section_number}")
            except Exception as e:
                logging.error(f"Error retrieving section documents: {str(e)}")
            
            # 2. Get case laws specific to this section
            case_law_query = f"landmark judgments supreme court section {section_number} IPC CrPC"
            try:
                case_results = vectorstore.similarity_search(
                    case_law_query,
                    k=3,
                    filter={"type": "case_study"}
                )
                combined_results.extend(case_results)
                logging.info(f"Retrieved {len(case_results)} case studies for Section {section_number}")
            except Exception as e:
                logging.error(f"Error retrieving case studies: {str(e)}")
                
        elif case_law_search:
            # Specialized search for case law queries
            logging.info("Case law specific query detected")
            try:
                case_results = vectorstore.similarity_search(
                    query,
                    k=4,
                    filter={"type": "case_study"}
                )
                combined_results.extend(case_results)
                logging.info(f"Retrieved {len(case_results)} case studies")
            except Exception as e:
                logging.error(f"Error retrieving case studies: {str(e)}")
        
        # Always add general search results
        try:
            general_results = vectorstore.similarity_search(
                query,
                k=3
            )
            # Only add general results that aren't duplicates
            for result in general_results:
                if result not in combined_results:
                    combined_results.append(result)
            logging.info(f"Retrieved {len(general_results)} general documents")
        except Exception as e:
            logging.error(f"Error retrieving general documents: {str(e)}")
        
        # Format the context string
        if combined_results:
            context = ""
            for i, doc in enumerate(combined_results):
                if hasattr(doc, 'page_content') and doc.page_content:
                    # Get metadata if available for better context
                    metadata_str = ""
                    if hasattr(doc, 'metadata') and doc.metadata:
                        metadata_str = f" (Source: {doc.metadata.get('source', 'Unknown')})"
                    
                    context += f"Document {i+1}{metadata_str}:\n{doc.page_content}\n\n"
            
            logging.info(f"Total context documents: {len(combined_results)}")
            return context
        else:
            logging.warning("No context documents found")
            return "No relevant context found."
        
    except Exception as e:
        logging.error(f"Error in context retrieval: {str(e)}")
        # Return empty context if retrieval fails
        return "No relevant context found."


def analyze_query_types(messages):
    """Analyze the types of queries in the chat history"""
    # Implement your query analysis logic here
    query_types = {"general": 0, "specific_law": 0, "case_reference": 0, "procedure": 0}
    
    for message in messages:
        if message.get("role") == "user":
            content = message.get("content", "").lower()
            if any(word in content for word in ["case", "judgment", "ruling"]):
                query_types["case_reference"] += 1
            elif any(word in content for word in ["section", "article", "act"]):
                query_types["specific_law"] += 1
            elif any(word in content for word in ["how to", "procedure", "process"]):
                query_types["procedure"] += 1
            else:
                query_types["general"] += 1
                
    return query_types


def store_feedback(username, feedback_type, feedback_text=None):
    """Store user feedback in the database"""
    conn = sqlite3.connect("app.db")
    c = conn.cursor()
    try:
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS feedback
            (username TEXT, feedback_type TEXT, feedback_text TEXT, timestamp REAL)
        """
        )
        c.execute(
            "INSERT INTO feedback (username, feedback_type, feedback_text, timestamp) VALUES (?, ?, ?, ?)",
            (username, feedback_type, feedback_text, time.time()),
        )
        conn.commit()
        return True
    except sqlite3.Error as e:
        logging.error(f"Error storing feedback: {str(e)}")
        return False
    finally:
        conn.close()


def store_analytics(username, analytics_data):
    """Store session analytics in the database"""
    conn = sqlite3.connect("app.db")
    c = conn.cursor()
    try:
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS analytics
            (username TEXT, session_duration REAL, message_count INTEGER, 
             query_types TEXT, timestamp REAL)
        """
        )
        c.execute(
            """INSERT INTO analytics 
               (username, session_duration, message_count, query_types, timestamp) 
               VALUES (?, ?, ?, ?, ?)""",
            (
                username,
             analytics_data["session_duration"],
             analytics_data["message_count"],
             json.dumps(analytics_data["query_types"]),
                analytics_data["timestamp"],
            ),
        )
        conn.commit()
        return True
    except sqlite3.Error as e:
        logging.error(f"Error storing analytics: {str(e)}")
        return False
    finally:
        conn.close()


def update_user_email_with_verification(username, new_email, password):
    """Update user email with password verification"""
    conn = sqlite3.connect("app.db")
    c = conn.cursor()
    try:
        # Debug logging for troubleshooting
        logging.info(f"Starting email update for user: {username}")
        
        # First verify the password
        c.execute("SELECT password FROM users WHERE username=?", (username,))
        stored_password = c.fetchone()
        
        if not stored_password:
            return (False, "User not found")
        
        if (
            stored_password[0] == password
        ):  # Direct comparison since password is already hashed
            # Check if email already exists
            c.execute(
                "SELECT username FROM users WHERE email=? AND username!=?",
                (new_email, username),
            )
            if c.fetchone():
                return (False, "Email already in use by another account")
            
            # Update email
            c.execute(
                "UPDATE users SET email=? WHERE username=?", (new_email, username)
            )
            conn.commit()
            return (True, "Email updated successfully")
        else:
            return (False, "Invalid password")
            
    except sqlite3.Error as e:
        logging.error(f"Error updating email: {str(e)}")
        return (False, f"Database error: {str(e)}")
    finally:
        conn.close()


def handle_login(username, password):
    hashed_password = hash_password(password)
    user = get_user(username, hashed_password)
    if user:
        logging.info(f"User login successful: {username}")
        logging.info(f"Profile image present: {bool(user['profile_image'])}")
        session_manager.create_session(
            username=username, profile_image=user["profile_image"], email=user["email"]
        )
        st.session_state.profile_image = user["profile_image"]
        st.session_state.email = user["email"]
        return True
    logging.warning(f"Login failed for user: {username}")
    return False


def create_sidebar(selected_tab):
    with st.sidebar:
        try:
            if not st.session_state.username:
                st.warning("Please log in to view previous chats")
                return
            
            # Add session duration indicator
            if "session_start_time" not in st.session_state:
                st.session_state.session_start_time = time.time()
                
            duration = int((time.time() - st.session_state.session_start_time) / 60)
            st.markdown(
                f"""
                <div style="background-color: #1E1E2D; padding: 8px 12px; border-radius: 8px; margin-bottom: 15px; font-size: 0.9em;">
                    ⏱️ {duration} mins
                </div>
                """,
                unsafe_allow_html=True
            )

            # Profile section
            st.markdown(
                f"""
                <div style="background-color: #1E1E2D; padding: 15px; border-radius: 8px; text-align: center; margin-bottom: 15px;">
                    <div style="width: 80px; height: 80px; margin: 0 auto; border-radius: 50%; overflow: hidden; border: 3px solid #2962FF;">
                        <img src="data:image/png;base64,{st.session_state.profile_image}" style="width: 100%; height: 100%; object-fit: cover;">
                    </div>
                    <div style="color: white; font-size: 1em; margin-top: 8px;">
                        {st.session_state.username}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            # Action buttons in a more compact layout
            col1, col2 = st.columns(2)
            with col1:
                if st.button("📤 Logout", key="logout_btn", use_container_width=True):
                    session_manager.end_session()
                    st.rerun()
            with col2:
                if st.button("➕ New", key="new_chat_btn", use_container_width=True):
                    st.session_state.current_session = None
                    st.session_state.messages = []
                    st.rerun()

            # Previous chats section with search
            st.markdown("## Previous Chats")
            search_term = st.text_input("🔍", key="chat_search", placeholder="Search chats...", label_visibility="collapsed")
            
            # Custom CSS for the chat cards
            st.markdown("""
                <style>
                    .chat-card {
                        background-color: #1E1E2F;
                        border-radius: 10px;
                        padding: 15px;
                        margin-bottom: 12px;
                        border: 1px solid rgba(255, 255, 255, 0.05);
                    }
                    .chat-title {
                        color: #E0E0E0;
                        font-size: 16px;
                        margin-bottom: 5px;
                        font-weight: 500;
                    }
                    .chat-meta {
                        color: #888;
                        font-size: 12px;
                        margin-bottom: 10px;
                    }
                    .error-card {
                        background-color: #703636;
                        color: white;
                        padding: 15px;
                        border-radius: 10px;
                        text-align: center;
                    }
                </style>
            """, unsafe_allow_html=True)
            
            # Get and display sessions
            try:
                sessions = list_sessions(st.session_state.username)
                
                if sessions:
                    for session in sessions:
                        if not search_term or search_term.lower() in session.get("session_name", "").lower():
                            session_name = session.get("session_name", "Unnamed Chat")
                            message_count = session.get("message_count", 0)
                            timestamp = datetime.fromisoformat(session.get("timestamp", datetime.now().isoformat())).strftime("%Y-%m-%d %H:%M")
                            filename = session.get("filename", "")
                            
                            # Display session info
                            st.markdown(f"""
                                <div class="chat-card">
                                    <div class="chat-title">{session_name}</div>
                                    <div class="chat-meta">{message_count} msgs • {timestamp}</div>
                                </div>
                            """, unsafe_allow_html=True)
                            
                            # Add buttons using simple columns
                            col1, col2 = st.columns(2)
                            with col1:
                                if st.button("Open", key=f"open_{filename}", use_container_width=True):
                                    try:
                                        session_path = os.path.join("sessions", filename)
                                        
                                        if os.path.exists(session_path):
                                            with open(session_path, "r", encoding="utf-8") as f:
                                                session_data = json.load(f)
                                            st.session_state.messages = session_data.get("messages", [])
                                            st.session_state.current_session = session_name
                                            st.rerun()
                                        else:
                                            st.error(f"Session file not found: {filename}")
                                    except Exception as e:
                                        logging.error(f"Error loading session: {str(e)}")
                                        st.error("Error loading chat session")
                            
                            with col2:
                                if st.button("Delete", key=f"delete_{filename}", use_container_width=True):
                                    try:
                                        session_path = os.path.join("sessions", filename)
                                        
                                        if os.path.exists(session_path):
                                            os.remove(session_path)
                                            if session_name == st.session_state.current_session:
                                                st.session_state.messages = []
                                                st.session_state.current_session = None
                                            st.success(f"Deleted chat: {session_name}")
                                            time.sleep(0.5)  # Brief pause to show success message
                                            st.rerun()
                                        else:
                                            st.error(f"Session file not found: {filename}")
                                    except Exception as e:
                                        logging.error(f"Error deleting session: {str(e)}")
                                        st.error("Error deleting chat session")
                else:
                    st.info("No previous chats")
                    
            except Exception as e:
                logging.error(f"Error loading sessions: {str(e)}")
                st.error(f"Error loading chats: {str(e)}")
                
        except Exception as e:
            logging.error(f"Error in create_sidebar: {str(e)}")
            st.error("Error creating sidebar")


def chat_interface():
    """Chat interface for both Chat and History sections"""
    try:
        # Initialize session state variables if they don't exist
        if "messages" not in st.session_state:
            st.session_state.messages = [
                {
                    "role": "assistant",
                    "content": "Hi, I'm AI-Vakeel, an AI Legal Advisor.",
                }
            ]
        
        if "username" not in st.session_state:
            st.session_state.username = None
        
        if "current_session" not in st.session_state:
            st.session_state.current_session = None

        # Add enhanced chat styling with dark theme
        st.markdown("""
        <style>
        /* Dark theme colors */
        :root {
            --bg-color: #1E1E1E;
            --text-color: #E0E0E0;
            --primary-color: #2962FF;
            --secondary-color: #0D47A1;
            --accent-color: #82B1FF;
            --hover-color: #1A237E;
        }

        .main {
            background-color: var(--bg-color);
            color: var(--text-color);
        }
        
        .chat-container {
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            background-color: var(--bg-color);
        }
        
        .stChatMessage {
            background: #2D2D2D;
            border-radius: 15px;
            padding: 15px;
            margin: 10px 0;
            box-shadow: 0 2px 5px rgba(0,0,0,0.2);
            transition: transform 0.2s ease;
            color: var(--text-color) !important;
        }
        
        .stChatMessage:hover {
            transform: translateY(-2px);
        }
        
        .stChatMessage[data-testid="user-message"] {
            background: linear-gradient(135deg, var(--primary-color) 0%, var(--secondary-color) 100%);
            color: white !important;
            margin-left: 20%;
        }
        
        .stChatMessage[data-testid="assistant-message"] {
            background: #2D2D2D;
            border-left: 4px solid var(--accent-color);
            margin-right: 20%;
            color: var(--text-color) !important;
        }
        
        .stChatInputContainer {
            padding: 10px;
            border-radius: 10px;
            background: #2D2D2D;
            box-shadow: 0 2px 5px rgba(0,0,0,0.2);
        }
        
        .stChatInput {
            border-radius: 20px !important;
            border: 2px solid var(--accent-color) !important;
            padding: 8px 15px !important;
            background-color: #363636 !important;
            color: var(--text-color) !important;
        }
        
        .stChatInput:focus {
            box-shadow: 0 0 0 2px rgba(130, 177, 255, 0.2) !important;
        }
        
        .thinking-animation {
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 20px;
            background-color: #2D2D2D;
            border-radius: 10px;
            margin: 10px 0;
        }

        /* Make all text in chat messages visible */
        .stChatMessage p, 
        .stChatMessage span, 
        .stChatMessage div {
            color: var(--text-color) !important;
        }

        /* Style markdown elements in messages */
        .stChatMessage h1, 
        .stChatMessage h2, 
        .stChatMessage h3 {
            color: var(--accent-color) !important;
        }

        .stChatMessage code {
            background-color: #363636 !important;
            color: #82B1FF !important;
            padding: 2px 5px;
            border-radius: 4px;
        }

        .stChatMessage a {
            color: var(--accent-color) !important;
            text-decoration: underline;
        }

        /* Style session name container */
        .session-container {
            background: linear-gradient(135deg, var(--primary-color) 0%, var(--secondary-color) 100%);
            color: white;
            padding: 10px 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            text-align: center;
            font-weight: 500;
        }
        </style>
        """, unsafe_allow_html=True)

        # Get user input
        prompt = st.chat_input("Ask your legal question...")

        # Only show session name input if no current session exists
        if not st.session_state.current_session:
            st.markdown("### Start New Conversation")
            session_input = st.text_input(
                "Enter a name for this conversation:", 
                placeholder="e.g., Property Law Query, Family Matter, etc.",
                key="new_session_name"
            )
            
            if session_input:
                try:
                    session_name = generate_session_name(str(session_input).strip())
                    if session_name:
                        st.session_state.current_session = session_name
                        st.rerun()
                except Exception as e:
                    logging.error(f"Error generating session name: {e}")
                    st.session_state.current_session = "New Chat"
                    st.rerun()

        # Auto-save session if conditions are met
        if (
            st.session_state.username
            and st.session_state.messages
            and st.session_state.current_session
        ):
            try:
                save_session(
                    username=st.session_state.username,
                    session_name=st.session_state.current_session,
                    session_data=st.session_state.messages
                )
            except Exception as e:
                logging.error(f"Error saving session: {e}")

        # Display current session name with enhanced styling
        if st.session_state.current_session:
            st.markdown(f"""
            <div class="session-container">
                Current Session: {st.session_state.current_session}
            </div>
            """, unsafe_allow_html=True)

        # Display chat messages with error handling for each message
        if st.session_state.messages:
            for message in st.session_state.messages:
                try:
                    if isinstance(message, dict) and "role" in message and "content" in message:
                        with st.chat_message(message["role"]):
                            st.markdown(message["content"])
                except Exception as e:
                    logging.error(f"Error displaying message: {e}")
                    continue

        # Handle user input
        if prompt:
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Get AI response with thinking animation
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                
                # Show thinking animation
                with st.spinner("AI-Vakeel is thinking..."):
                    thinking_lottie = load_lottie_url(THINKING_ANIMATION_URL)
                    if thinking_lottie:
                        with message_placeholder.container():
                            st_lottie(
                                thinking_lottie,
                                key=f"thinking_{len(st.session_state.messages)}",
                                height=100,
                                speed=1,
                                loop=True
                            )
                    
                    # Get response
                    response = user_input(prompt)
                    
                    if response:
                        # Clear thinking animation
                        message_placeholder.empty()
                        
                        # Save to session and display
                        st.session_state.messages.append({"role": "assistant", "content": response})
                        
                        # Save chat to database
                        try:
                            if st.session_state.username and st.session_state.current_session:
                                add_chat(st.session_state.username, "user", prompt)
                                add_chat(st.session_state.username, "assistant", response)
                                
                                # Save session
                                save_session(
                                    username=st.session_state.username,
                                    session_name=st.session_state.current_session,
                                    session_data=st.session_state.messages
                                )
                        except Exception as e:
                            logging.error(f"Error saving chat: {e}")
                        
                        # Display response with typing effect
                        display_typing_effect(response)

        # Auto-save session periodically
        if (
            st.session_state.username 
            and st.session_state.messages 
            and st.session_state.current_session
        ):
            try:
                save_session(
                    username=st.session_state.username,
                    session_name=st.session_state.current_session,
                    session_data=st.session_state.messages
                )
            except Exception as e:
                logging.error(f"Error auto-saving session: {e}")

    except Exception as e:
        logging.error(f"Error in chat interface: {e}")
        st.error("Error displaying chat messages. Please refresh the page.")


def verify_vector_store():
    """Verify that the vector store is properly loaded with data"""
    global vectorstore
    
    try:
        logging.info("Testing vector store...")
        
        if vectorstore is None:
            # Try to reinitialize if not available
            embeddings = GoogleGenerativeAIEmbeddings(
                google_api_key=GOOGLE_API_KEY,
                model="models/embedding-001"
            )
            vectorstore = FAISS.load_local(folder_path="Faiss", embeddings=embeddings)
            logging.info("Vector store reinitialized successfully")
        
        # Test a simple query
        results = vectorstore.similarity_search("test query", k=1)
        logging.info(f"Search successful, found {len(results)} results")
        return True
            
    except Exception as e:
        logging.error(f"Error testing vector store: {str(e)}")
        return False


def initialize_embeddings():
    """Initialize embeddings with error handling"""
    try:
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",  # Changed from text-embedding-004
            google_api_key=GOOGLE_API_KEY,
        )
        
        # Test if vector store can be loaded
        try:
            # Removed allow_dangerous_deserialization
            vectorstore = FAISS.load_local(folder_path="Faiss", embeddings=embeddings)
            logging.info("Vector store loaded successfully")
            return embeddings, vectorstore
        except Exception as e:
            logging.error(f"Error loading vector store: {str(e)}")
            # Try to recover by rebuilding dummy database
            try:
                from rebuild_legal_db import create_dummy_database

                if create_dummy_database():
                    logging.info("Rebuilt dummy database, trying again")
                    try:
                        vectorstore = FAISS.load_local(
                            folder_path="Faiss", embeddings=embeddings
                        )
                        logging.info("Vector store loaded successfully after rebuild")
                        return embeddings, vectorstore
                    except Exception as e2:
                        logging.error(
                            f"Still can't load vector store after rebuild: {str(e2)}"
                        )
            except Exception as rebuilder_error:
                logging.error(f"Rebuild attempt failed: {str(rebuilder_error)}")
            
            # Return None if everything fails
            return embeddings, None
            
    except Exception as e:
        logging.error(f"Error initializing embeddings: {str(e)}")
        return None, None


def main():
    try:
        # Get PDF files from both subdirectories
        pdf_docs = []

        # Check case_studies directory
        case_studies_dir = os.path.join("dataset", "case_studies")
        if os.path.exists(case_studies_dir):
            for file in os.listdir(case_studies_dir):
                if file.endswith(".pdf"):
                    pdf_docs.append(os.path.join(case_studies_dir, file))
                    logging.info(f"Found case study: {file}")

        # Check legal_documents directory
        legal_docs_dir = os.path.join("dataset", "legal_documents")
        if os.path.exists(legal_docs_dir):
            for file in os.listdir(legal_docs_dir):
                if file.endswith(".pdf"):
                    pdf_docs.append(os.path.join(legal_docs_dir, file))
                    logging.info(f"Found legal document: {file}")
    except Exception as e:
        logging.error(f"Error getting PDF files: {str(e)}")
        st.error("Failed to load PDF files. Please try again.")

    # Update custom CSS
    st.markdown(
        """
        <style>
        .main-title {
            text-align: center;
            font-size: 3.2em;
            color: #1E3A8A;
            margin-bottom: 0;
            padding-bottom: 0;
        }
        .subtitle {
            text-align: center;
            font-size: 1.5em;
            color: #4B5563;
            margin-top: 0;
            padding-top: 0;
            margin-bottom: 2em;
        }
        .login-container {
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 2em;
            max-width: 1200px;
            margin: 0 auto;
        }
        .login-box {
            background: transparent !important;
            padding: 2em;
            border-radius: 10px;
            width: 100%;
            max-width: 400px;
        }
        .stButton>button {
            width: 100%;
            background-color: #1E3A8A;
            color: white;
        }
        div[data-testid="stSubheader"] {
            color: #1E3A8A;
            background: transparent !important;
        }
        .sidebar-container {
            background: linear-gradient(180deg, #1a237e 0%, #0d47a1 100%);
            border-radius: 15px;
            padding: 20px;
            color: white;
            margin: -1rem -1rem 1rem -1rem;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            position: relative;
            padding-top: 80px;
        }
        
        .profile-header {
            display: flex;
            flex-direction: column;
            align-items: center;
            padding: 20px 0;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
            margin-bottom: 20px;
            position: relative;
        }
        
        .profile-image-wrapper {
            width: 120px;
            height: 120px;
            position: absolute;
            top: -60px;
            left: 50%;
            transform: translateX(-50%);
            margin-bottom: 15px;
            z-index: 1;
        }
        
        .profile-image {
            width: 100%;
            height: 100%;
            border-radius: 50%;
            overflow: hidden;
            border: 4px solid white;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
            background-color: white;
        }
        
        .profile-image img {
            width: 100%;
            height: 100%;
            object-fit: cover;
        }
        
        .username {
            font-size: 1.2rem;
            font-weight: 600;
            color: white;
            text-align: center;
            margin-top: 10px;
        }
        
        .chat-session {
            background: rgba(255, 255, 255, 0.1);
            padding: 12px;
            border-radius: 8px;
            margin-bottom: 8px;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        
        .chat-session:hover {
            background: rgba(255, 255, 255, 0.2);
            transform: translateX(5px);
        }
        
        .session-title {
            font-weight: 500;
            margin-bottom: 4px;
        }
        
        .session-meta {
            font-size: 0.8rem;
            opacity: 0.8;
        }
        
        .session-buttons {
            display: flex;
            gap: 5px;
            margin-top: 5px;
        }
        
        .sidebar-button {
            width: 100%;
            background: rgba(255, 255, 255, 0.1);
            border: 1px solid rgba(255, 255, 255, 0.2);
            color: white;
            padding: 8px;
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        
        .sidebar-button:hover {
            background: rgba(255, 255, 255, 0.2);
            transform: translateY(-2px);
        }
        </style>
            """,
        unsafe_allow_html=True,
    )

    # Check authentication status
    if not st.session_state.get("authentication_status", False):
        # Display title and subtitle
        st.markdown('<h1 class="main-title">AI-Vakeel</h1>', unsafe_allow_html=True)
        st.markdown(
            '<p class="subtitle">Your Intelligent Legal Assistant</p>',
            unsafe_allow_html=True,
        )
        
        # Create two columns for image and login
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # Replace use_column_width with use_container_width
            st.image("loginimg.png", use_container_width=True)  # Updated parameter
        
        with col2:
            menu = ["Login", "Sign Up"]
            choice = st.selectbox(
                label="Authentication Choice",  # Proper label for accessibility
                options=menu,
                label_visibility="collapsed",  # Hides the label while maintaining accessibility
            )
            
            st.markdown('<div class="login-container">', unsafe_allow_html=True)
            if choice == "Login":
                st.markdown('<div class="login-box">', unsafe_allow_html=True)
                st.subheader("Welcome Back! 👋")
                username = st.text_input("Username", key="login_username")
                password = st.text_input(
                    "Password", type="password", key="login_password"
                )
                if st.button("Login", key="login_button"):
                    if handle_login(username, password):
                        st.rerun()
                    else:
                        st.error("Username/password is incorrect")
                st.markdown("</div>", unsafe_allow_html=True)
            
            elif choice == "Sign Up":
                st.markdown('<div class="login-box">', unsafe_allow_html=True)
                st.subheader("Create Account ✨")
                new_username = st.text_input("Username", key="signup_username")
                new_password = st.text_input(
                    "Password", type="password", key="signup_password"
                )
                confirm_password = st.text_input(
                    "Confirm Password", type="password", key="signup_confirm"
                )
                uploaded_file = st.file_uploader(
                    "Profile Image", type=["png", "jpg", "jpeg"], key="signup_image"
                )
                
                if st.button("Sign Up", key="signup_button"):
                    if new_password == confirm_password:
                        hashed_password = hash_password(new_password)
                        if uploaded_file is not None:
                            image_bytes = uploaded_file.read()
                            profile_image = base64.b64encode(image_bytes).decode()
                        else:
                            profile_image = None
                        success = add_user(new_username, hashed_password, profile_image)
                        if success:
                            st.success("Account created successfully!")
                        else:
                            st.error("Username already exists")
                    else:
                        st.error("Passwords do not match")
                st.markdown("</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
    else:
        # User is authenticated
        # Navigation menu
        selected = option_menu(
            menu_title=None,
            options=["Chat", "History", "Profile"],
            icons=["chat-dots-fill", "clock-history", "person-circle"],
            orientation="horizontal",
        )
        
        # Create sidebar with current tab
        create_sidebar(selected)
        
        if selected == "Chat":
            # Remove the duplicate chat input code from here
            chat_interface()  # This will handle all chat functionality

            # Keep the debug section
            if st.checkbox("Show Available Dummy Data (Debug)"):
                st.subheader("Available Dummy Data Types")
                debug_data_type = st.selectbox(
                    "Select data type to view:",
                    [
                        "Court Vacancy",
                        "Case Status",
                        "Traffic Violations",
                        "Fast Track Courts",
                        "Live Streams",
                    ],
                )

                if debug_data_type == "Court Vacancy":
                    st.json(get_court_vacancy())
                elif debug_data_type == "Case Status":
                    case_id = get_random_case_number()
                    st.json(get_case_status(case_id))
                elif debug_data_type == "Traffic Violations":
                    violation_id = get_random_violation_id()
                    st.json(get_traffic_violation(violation_id))
                elif debug_data_type == "Fast Track Courts":
                    st.json(get_fast_track_courts())
                elif debug_data_type == "Live Streams":
                    st.json(get_live_streams())
        
        elif selected == "History":
            st.header("Chat History")
            # Display chat history without creating new input
            if "messages" in st.session_state:
                for message in st.session_state.messages:
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"])
            
            # Add download button for chat history
            if st.session_state.messages and len(st.session_state.messages) > 1:
                st.download_button(
                    "📥 Download Chat History",
                    data=json.dumps(st.session_state.messages, indent=2),
                    file_name="chat_history.json",
                    mime="application/json",
                )
        
        elif selected == "Profile":
            st.header("Profile Settings")
            
            # Profile Image Section
            col1, col2 = st.columns([1, 2])
            with col1:
                if st.session_state.profile_image:
                    st.image(
                        f"data:image/png;base64,{st.session_state.profile_image}",
                        width=200,
                        caption="Current Profile Picture",
                    )
                else:
                    st.info("No profile picture uploaded")
            
            with col2:
                new_profile_image = st.file_uploader(
                    "Update Profile Picture",
                                                   type=["jpg", "jpeg", "png"], 
                    key="profile_update",
                )
                if new_profile_image and st.button(
                    "Update Picture", key="update_pic_btn"
                ):
                    try:
                        image_bytes = new_profile_image.read()
                        profile_image = base64.b64encode(image_bytes).decode()
                        if update_profile_image(
                            st.session_state.username, profile_image
                        ):
                            # Verify the image was stored correctly
                            stored_image = verify_profile_image(
                                st.session_state.username
                            )
                            if stored_image:
                                st.session_state.profile_image = stored_image
                                st.success("Profile picture updated successfully!")
                                st.rerun()
                            else:
                                st.error("Profile picture was not stored correctly")
                        else:
                            st.error("Failed to update profile picture")
                    except Exception as e:
                        st.error(f"Error updating profile picture: {str(e)}")
                        logging.error(f"Profile update error: {str(e)}")
            
            # Account Settings
            st.subheader("Account Settings")
            with st.expander("Update Email"):
                # Show current email
                current_email = get_user_email(st.session_state.username)
                if current_email:
                    st.info(f"Current Email: {current_email}")
                else:
                    st.info("No email set")
                
                new_email = st.text_input("New Email Address", key="new_email")
                verify_password = st.text_input(
                    "Verify Password", type="password", key="verify_pass_email"
                )
                
                if st.button("Update Email", key="update_email_btn"):
                    if new_email and verify_password:
                        # Add debug logging
                        logging.info(
                            f"Attempting to update email for user: {st.session_state.username}"
                        )
                        
                        # Hash the password before verification
                        hashed_password = hash_password(verify_password)
                        
                        try:
                            success, message = update_user_email_with_verification(
                                st.session_state.username, new_email, hashed_password
                            )
                            
                            if success:
                                st.success(message)
                                # Update session state with new email
                                st.session_state.email = new_email
                                time.sleep(
                                    1
                                )  # Give time for the success message to be seen
                                st.rerun()
                            else:
                                st.error(message)
                                logging.warning(f"Email update failed: {message}")
                        except Exception as e:
                            st.error("Failed to update email. Please try again.")
                            logging.error(f"Email update error: {str(e)}")
                        else:
                            st.warning("Please fill in all fields")
            
            with st.expander("Change Password"):
                col3, col4 = st.columns(2)
                with col3:
                    current_password = st.text_input(
                        "Current Password", type="password", key="current_pass"
                    )
                    new_password = st.text_input(
                        "New Password", type="password", key="new_pass"
                    )
                with col4:
                    confirm_new_password = st.text_input(
                        "Confirm New Password", type="password", key="confirm_new_pass"
                    )

                if st.button("Change Password", key="change_pass_btn"):
                    if current_password and new_password and confirm_new_password:
                        if new_password != confirm_new_password:
                            st.error("New passwords do not match")
                        else:
                            if update_password(
                                st.session_state.username,
                                hash_password(current_password),
                                hash_password(new_password),
                            ):
                                st.success("Password updated successfully!")
                                if st.button("Logout Now"):
                                    session_manager.end_session()
                                st.rerun()
                            else:
                                st.error(
                                    "Failed to update password. Please check your current password."
                                )
                    else:
                        st.warning("Please fill in all password fields")
            
            # Danger Zone
            st.subheader("Danger Zone", help="Careful! These actions cannot be undone.")
            with st.expander("Delete Account"):
                st.warning("⚠️ This action is permanent and cannot be undone!")
                delete_password = st.text_input(
                    "Enter your password to confirm", type="password", key="delete_pass"
                )
                confirm_delete = st.checkbox(
                    "I understand that this action cannot be undone",
                    key="confirm_delete",
                )

                if st.button(
                    "Delete My Account", type="primary", key="delete_account_btn"
                ):
                    if delete_password and confirm_delete:
                        if delete_user_account(
                            st.session_state.username, delete_password
                        ):
                            session_manager.end_session()
                            st.success("Account deleted successfully")
                            st.rerun()
                        else:
                            st.error(
                                "Failed to delete account. Please check your password."
                            )
                    else:
                        st.warning("Please enter your password and confirm the action")
            
            # Account Statistics
            st.subheader("Account Statistics")
            col5, col6, col7 = st.columns(3)
            with col5:
                total_chats = len(list_sessions(st.session_state.username))
                st.metric("Total Conversations", total_chats)
            with col6:
                # Add this function to db.py if not already present
                total_messages = get_total_messages(st.session_state.username)
                st.metric("Total Messages", total_messages)
            with col7:
                account_age = get_account_age(
                    st.session_state.username
                )  # Add this function
                st.metric("Account Age", f"{account_age} days")

    # Add this check in the main function
    if not verify_vector_store():
        st.error("Error: Legal database not properly loaded. Please contact support.")

    # Add this function to automatically name sessions based on first question
    if not st.session_state.current_session and len(st.session_state.messages) == 0:
        # This is the first message of a new session
        if "prompt" in locals() and isinstance(prompt, str):
            session_name = generate_session_name(prompt)
            st.session_state.current_session = session_name
        else:
            st.session_state.current_session = "New Chat"


def validate_legal_response(response):
    """Validate and enhance the legal response"""
    try:
        # Check if response contains case studies section
        has_case_studies = "# Relevant Case Law" in response or "Case Law" in response
        
        validation_prompt = f"""
        Verify this legal response for accuracy, completeness, and proper formatting:
        
        RESPONSE TO VALIDATE:
        {response}
        
        Check for:
        1. Specific legal citations:
           - Acts should include section numbers
           - Case laws should have proper citations (including court and year)
           - Recent amendments should be mentioned
        
        2. Legal accuracy:
           - Correctness of cited sections
           - Accuracy of case law references
           - Current validity of laws cited
        
        3. Completeness:
           - All relevant laws covered
           - Practical guidance provided
           - Clear next steps outlined
           
        4. Case Studies Analysis:
           - Are landmark cases included with full citations?
           - Is the analysis of each case thorough and relevant?
           - Are the principles from each case clearly extracted?
           - Is the relevance to the query explained?
        
        Be particularly critical about case study citations and analysis.
        """
        
        model = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0.1,
            google_api_key=GOOGLE_API_KEY
        )
        
        validation_result = model.invoke(validation_prompt)
        validation_text = validation_result.content if hasattr(validation_result, 'content') else str(validation_result)
        
        # Check for specific issues
        issues = []
        
        # Case study issues
        if has_case_studies and any(term in validation_text.lower() for term in ["missing citation", "incomplete citation", "lack of case", "no case studies"]):
            issues.append("Case Studies: Missing or incomplete case citations")
        
        # Legal accuracy issues
        if any(term in validation_text.lower() for term in ["incorrect", "inaccurate", "wrong", "error"]):
            issues.append("Legal Accuracy: Incorrect legal references or information")
        
        # Completeness issues
        if any(term in validation_text.lower() for term in ["incomplete", "missing", "lacks", "should include"]):
            issues.append("Completeness: Missing essential legal information")
        
        if issues:
            logging.warning(f"Response validation found issues: {', '.join(issues)}")
            
            # Try to enhance the response with more detailed case studies
            if "Case Studies" in issues[0]:
                try:
                    enhancement_prompt = f"""
                    The following legal response lacks proper case studies or has incomplete case citations:
                    
                    {response}
                    
                    Please enhance ONLY the case studies section with:
                    1. At least 2-3 landmark cases with COMPLETE citations (case name, citation, court, year)
                    2. Brief facts of each case
                    3. Key principles established by each case
                    4. Relevance to the query
                    
                    Keep the other parts of the response unchanged.
                    """
                    
                    enhanced_result = model.invoke(enhancement_prompt)
                    enhanced_response = enhanced_result.content if hasattr(enhanced_result, 'content') else str(enhanced_result)
                    
                    logging.info("Enhanced response with better case studies")
                    return True, enhanced_response
                    
                except Exception as e:
                    logging.error(f"Error enhancing case studies: {str(e)}")
            
            # If enhancement fails or other issues, return original response with warnings
            return True, f"""
{response}

---
*Note: This response may have some limitations in the case studies cited or legal details provided.*
"""
        
        return True, response
        
    except Exception as e:
        logging.error(f"Validation error: {str(e)}")
        # On validation error, return True to allow the response through
        return True, response


# Add this function to handle DoJ specific queries
def handle_doj_query(query: str) -> Optional[str]:
    """Handle Department of Justice related queries with dummy data responses"""
    query_lower = query.lower()

    # If query is about articles, laws, or sections, return None to let Gemini handle it
    if any(
        term in query_lower
        for term in ["article", "section", "law", "ipc", "crpc", "constitution"]
    ):
        return None

    # Court vacancy queries
    if any(
        term in query_lower
        for term in ["court vacancy", "vacancies", "judge", "judicial"]
    ):
        from data.dummy_data import get_court_vacancy, match_state_from_query

        try:
            # Extract state name from query
            state = match_state_from_query(query)
            vacancy_data = get_court_vacancy(state)

            # Format response
            response = "**Court Vacancy Information**\n\n"
            response += "*Using AI-Vakeel's dummy data*\n\n"

            # Supreme Court details
            sc = vacancy_data["supreme_court"]
            response += "**Supreme Court of India**\n"
            response += (
                f"- Total sanctioned strength: {sc.get('total_judges', 'N/A')}\n"
            )
            response += f"- Current judges: {sc.get('current_judges', 'N/A')}\n"
            response += f"- Vacancies: {sc.get('vacancies', 'N/A')}\n"
            response += f"- Chief Justice: {sc.get('chief_justice', 'N/A')}\n\n"

            # High Courts details (specific state or all)
            if state and vacancy_data["high_courts"]:
                hc = vacancy_data["high_courts"][0]
                response += f"**{hc.get('name', 'High Court')}**\n"
                response += (
                    f"- Total sanctioned strength: {hc.get('total_judges', 'N/A')}\n"
                )
                response += f"- Current judges: {hc.get('current_judges', 'N/A')}\n"
                response += f"- Vacancies: {hc.get('vacancies', 'N/A')}\n"
                response += f"- Chief Justice: {hc.get('chief_justice', 'N/A')}\n\n"

            # District Courts summary if state is specified
            if state and vacancy_data["district_courts"]:
                dc = vacancy_data["district_courts"][0]
                response += f"**District Courts in {state}**\n"
                response += (
                    f"- Total sanctioned posts: {dc.get('total_judges', 'N/A')}\n"
                )
                response += f"- Current judges: {dc.get('current_judges', 'N/A')}\n"
                response += f"- Vacancies: {dc.get('vacancies', 'N/A')}\n\n"

            response += f"*Last updated: {sc.get('last_updated', 'Unknown')}*"
            return response
        except Exception as e:
            logging.error(f"Error in court vacancy handler: {str(e)}")
            return "I'm sorry, I couldn't retrieve the court vacancy information at this time. Please try again later."

    # Case status queries
    elif any(
        term in query_lower
        for term in ["case status", "status of case", "hearing", "pending case"]
    ) or re.search(r"([A-Z]+-\d{4}-\d{3})", query):
        try:
            # Extract case number if present
            case_match = re.search(r"([A-Z]+-\d{4}-\d{3})", query)

            from data.dummy_data import get_case_status, get_random_case_number

            # Use case number from query, or random one if not specified
            case_number = None
            if case_match:
                case_number = case_match.group(1)
            else:
                # If asking about next hearing but no specific case, use a random one
                case_number = get_random_case_number()

            # Get case data
            case_data = get_case_status(case_number)

            # Format response
            response = f"**Case Status: {case_number}**\n\n"
            response += "*Using AI-Vakeel's dummy data*\n\n"

            if isinstance(case_data, dict):  # Single case
                response += f"**Court:** {case_data.get('court', 'N/A')}\n"
                response += f"**Status:** {case_data.get('status', 'N/A')}\n"
                response += f"**Filing Date:** {case_data.get('filing_date', 'N/A')}\n"
                response += (
                    f"**Next Hearing:** {case_data.get('next_hearing', 'N/A')}\n"
                )
                response += f"**Judge:** {case_data.get('judge', 'N/A')}\n"
                response += f"**Category:** {case_data.get('category', 'N/A')}\n\n"

                # Add petitioner and respondent if available
                if "petitioner" in case_data:
                    response += (
                        f"**Petitioner:** {case_data.get('petitioner', 'N/A')}\n"
                    )
                if "respondent" in case_data:
                    response += (
                        f"**Respondent:** {case_data.get('respondent', 'N/A')}\n\n"
                    )

                # Add last hearing and order if available
                if "last_hearing" in case_data:
                    response += (
                        f"**Last Hearing:** {case_data.get('last_hearing', 'N/A')}\n"
                    )
                if "last_order" in case_data:
                    response += (
                        f"**Last Order:** {case_data.get('last_order', 'N/A')}\n\n"
                    )

                # Add options for next steps if asking about a hearing
                if "next hearing" in query_lower or "hearing" in query_lower:
                    response += "**Would you like to:**\n"
                    response += "1. Set a reminder for the next hearing date?\n"
                    response += "2. Get tips for preparing for the hearing?\n"
                    response += (
                        "3. Learn about possible outcomes of the next hearing?\n"
                    )

            else:  # No case found
                response += "Case not found. Please verify the case number."

            return response
        except Exception as e:
            logging.error(f"Error in case status handler: {str(e)}")
            return "I'm sorry, I couldn't retrieve the case status information at this time. Please try again later."

    # Traffic violation queries
    elif any(
        term in query_lower for term in ["traffic", "challan", "violation", "fine"]
    ) or re.search(r"(TV-\d{4}-\d{3})", query_lower):
        try:
            from data.dummy_data import get_traffic_violation, get_random_violation_id

            # Extract violation ID if present
            violation_match = re.search(r"(TV-\d{4}-\d{3})", query)
            violation_id = (
                violation_match.group(1)
                if violation_match
                else get_random_violation_id()
            )

            violation_data = get_traffic_violation(violation_id)

            response = "**Traffic Violation Information**\n\n"
            response += "*Using AI-Vakeel's dummy data*\n\n"

            if isinstance(violation_data, dict):
                response += f"**Violation ID:** {violation_id}\n"
                response += f"**Type:** {violation_data.get('type', 'N/A')}\n"
                response += f"**Location:** {violation_data.get('location', 'N/A')}\n"
                response += f"**Date:** {violation_data.get('date', 'N/A')}\n"
                response += (
                    f"**Fine Amount:** {violation_data.get('fine_amount', 'N/A')}\n"
                )
                response += f"**Status:** {violation_data.get('status', 'N/A')}\n"
            else:
                response += (
                    "Violation record not found. Please verify the violation ID."
                )

            return response

        except Exception as e:
            logging.error(f"Error in traffic violation handler: {str(e)}")
            return "I'm sorry, I couldn't retrieve the traffic violation information at this time. Please try again later."
    
    return None  # Return None for all other queries to let Gemini handle them


def get_articles_overview():
    """Get comprehensive overview of legal articles"""
    return """
Key Legal Articles and Sections:

1. Fundamental Rights:
   - Article 14: Right to Equality
   - Article 15: Prohibition of Discrimination
   - Article 19: Right to Freedom
   - Article 21: Right to Life and Liberty
   - Article 32: Right to Constitutional Remedies

2. Criminal Law (IPC):
   - Sections 299-304: Homicide and Murder
   - Sections 375-376: Sexual Offences
   - Sections 317-318: Offences Against Children
   - Sections 378-382: Theft and Robbery
   - Sections 405-409: Criminal Breach of Trust

3. Civil Law (CPC):
   - Section 9: Civil Courts Jurisdiction
   - Section 80: Notice to Government
   - Section 89: Settlement of Disputes
   - Section 100: Second Appeal
   - Section 151: Inherent Powers of Court

Please specify which article or section you'd like to know more about.
"""


# Add this function to enhance response generation
def get_enhanced_response(query, context=None):
    """Get enhanced response using both trained data and internet context"""
    try:
        # Check for greetings first
        if is_greeting(query):
            model = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash",
                temperature=0.7,
                google_api_key=GOOGLE_API_KEY
            )
            
            greeting_prompt = """
            As AI-Vakeel, provide a warm and professional greeting. Introduce yourself as an AI legal assistant and explain your capabilities in helping with:

            1. Legal Information and Research:
               - Constitutional articles and their interpretations
               - IPC and CrPC sections
               - Recent legal amendments
               - Supreme Court judgments
            
            2. Case Law and Precedents:
               - Finding relevant case studies
               - Analyzing landmark judgments
               - Understanding legal principles
            
            3. Legal Procedures:
               - Step-by-step guidance
               - Documentation requirements
               - Filing procedures
               - Timeline information
            
            4. Rights and Remedies:
               - Understanding fundamental rights
               - Available legal remedies
               - Procedural requirements
               - Approaching authorities

            Make the response warm, professional, and inviting while maintaining accuracy about legal capabilities.
            """
            
            try:
                greeting_response = model.invoke(greeting_prompt)
                return greeting_response.content if hasattr(greeting_response, 'content') else str(greeting_response)
            except Exception as e:
                logging.error(f"Error generating greeting: {str(e)}")
                return "Hello! I'm AI-Vakeel, your legal assistant. How can I help you today?"

        # For non-greeting queries, continue with regular processing
        legal_context = context if context else ""
        case_studies_context = ""

        # Try to get case studies from vector store first
        try:
            if vectorstore:
                # Search specifically for case law and precedents
                case_study_query = f"case law precedents judgments related to {query}"
                case_studies = vectorstore.similarity_search(case_study_query, k=3)
                if case_studies:
                    case_studies_context = "\n\n".join([doc.page_content for doc in case_studies])
                    logging.info(f"Found {len(case_studies)} relevant case studies in vector store")
        except Exception as e:
            logging.warning(f"Error retrieving case studies from vector store: {str(e)}")

        model = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0.7,
            google_api_key=GOOGLE_API_KEY,
            generation_config={
                "max_output_tokens": 4096,
                "top_p": 0.95,
                "top_k": 40
            }
        )

        # Create main legal response prompt
        legal_prompt = f"""
        As AI-Vakeel, provide a comprehensive legal response for:
        
        QUERY: {query}
        
        Use the following context information:
        {legal_context}
        
        Structure your response in this format:
        
        1. APPLICABLE LAWS AND PROVISIONS:
           - Constitutional Articles (with exact numbers)
           - IPC Sections (with exact numbers)
           - CrPC Sections (with exact numbers)
           - Other relevant statutes
           - Recent amendments or changes
        
        2. LEGAL INTERPRETATION:
           - Key legal principles
           - Supreme Court interpretations
           - Current legal position
           - Important definitions
        
        3. RIGHTS AND REMEDIES:
           - Fundamental rights involved
           - Legal remedies available
           - Procedural requirements
           - Timeline considerations
           - Authorities to approach

        4. PRACTICAL GUIDANCE:
           - Step-by-step procedure
           - Required documentation
           - Common challenges
           - Best practices
           - Preventive measures

        Include all relevant legal citations and references.
        Make sure all article numbers and section numbers are accurate.
        Focus on practical, actionable guidance for Indian law.
        """

        # Get the main legal response
        legal_response = model.invoke(legal_prompt)

        # Create case studies prompt
        case_studies_prompt = f"""
        Based on the following context and your knowledge, provide detailed analysis of relevant case law and precedents for:
        
        QUERY: {query}
        
        {f"AVAILABLE CASE STUDIES:\n{case_studies_context}" if case_studies_context else "Please use your knowledge to provide relevant case studies and precedents."}
        
        Structure your response to include:
        1. LANDMARK CASES:
           - Case names with citations
           - Key facts and issues
           - Court's reasoning
           - Principles established
        
        2. RECENT PRECEDENTS:
           - Recent relevant judgments
           - Current interpretation
           - How they apply to this query
        
        3. PRACTICAL IMPLICATIONS:
           - How these cases affect similar situations
           - Important distinctions or exceptions
           - Current legal position
        
        Focus on Indian case law and ensure all citations are accurate.
        If using cases from vector store, expand on their implications.
        If no specific cases are available, provide analysis of most relevant precedents from your knowledge.
        """
            
        # Get case studies analysis
        try:
            case_studies_response = model.invoke(case_studies_prompt)
            logging.info("Successfully generated case studies response")
        except Exception as e:
            logging.error(f"Error generating case studies response: {str(e)}")
            case_studies_response = None

        # Combine responses with proper formatting
        if case_studies_response and hasattr(case_studies_response, 'content'):
            final_response = f"""
# Legal Analysis

{legal_response.content}

# Relevant Case Studies and Precedents

{case_studies_response.content}

*This response is based on Indian law and relevant legal precedents. For specific legal advice, please consult a qualified lawyer.*
"""
        else:
            final_response = f"""
# Legal Analysis

{legal_response.content}

*This response is based on Indian law. For specific legal advice, please consult a qualified lawyer.*
"""
        
        # Validate the response
        is_valid, enhanced_response = validate_legal_response(final_response)
        return enhanced_response if is_valid else get_focused_response(query, context)
            
    except Exception as e:
        logging.error(f"Error in enhanced response generation: {str(e)}")
        return "I apologize, but I'm having trouble generating a complete response. Please try again."


def get_relevant_case_studies(query):
    """Get relevant case studies for a query"""
    try:
        # Initialize embeddings
        embeddings, vectorstore = initialize_embeddings()
        
        if not vectorstore:
            logging.warning("No vector store available for case studies")
            return ""
            
        # Search for case studies
        try:
            results = vectorstore.similarity_search(
                query, k=2, filter={"type": "case_study"}
            )
            
            if not results:
                return ""
                
            case_studies = "\n\n".join(
                [
                f"Case: {doc.page_content}"
                    for doc in results
                    if hasattr(doc, "page_content")
                ]
            )
            
            return case_studies
        except Exception as e:
            logging.error(f"Error searching for case studies: {str(e)}")
            return ""
            
    except Exception as e:
        logging.error(f"Error retrieving case studies: {e}")
        return ""


def get_focused_response(query, context):
    """Get a more focused response when initial response validation fails"""
    try:
        model = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=0.3,  # Lower temperature for more focused response
            google_api_key=GOOGLE_API_KEY,
        )
        
        focused_prompt = f"""
        Provide a precise legal response for:
        Query: {query}
        Context: {context}
        
        Focus on:
        1. Exact legal provisions
        2. Current valid laws
        3. Verified information only
        """
        
        response = model.invoke(focused_prompt)
        return response.content
        
    except Exception as e:
        logging.error(f"Error in focused response: {str(e)}")
        return "I apologize, but I'm having trouble accessing the legal information. Please try again."


def handle_profile_image_upload(uploaded_file):
    """Handle profile image upload with validation"""
    try:
        if uploaded_file is None:
            return None
            
        # Verify file size (max 5MB)
        if uploaded_file.size > 5 * 1024 * 1024:
            st.error("File size too large. Maximum size is 5MB.")
            return None
            
        # Verify file type
        allowed_types = ["image/jpeg", "image/png"]
        if uploaded_file.type not in allowed_types:
            st.error("Invalid file type. Please upload JPEG or PNG images only.")
            return None
            
        # Read and encode image
        image_bytes = uploaded_file.read()
        encoded_image = base64.b64encode(image_bytes).decode()
        
        # Verify encoded image
        try:
            base64.b64decode(encoded_image)
            return encoded_image
        except:
            st.error("Error processing image. Please try another image.")
            return None
            
    except Exception as e:
        logging.error(f"Profile image upload error: {str(e)}")
        st.error("Error uploading image. Please try again.")
        return None
            
            
def generate_session_name(input_text):
    """Generate a session name from user input"""
    try:
        # Type checking
        if not isinstance(input_text, str):
            error_msg = f"Invalid input type for session name generation: {type(input_text)}"
            logging.warning(error_msg)
            return "New Chat"
        
        # Clean input
        input_text = input_text.strip()
        if not input_text:
            logging.info("Empty input text for session name, using default")
            return "New Chat"
        
        # Clean and truncate the question
        clean_question = re.sub(r"[^\w\s]", "", input_text)
        words = clean_question.split()
        
        # For greetings, return default name
        greeting_words = {"hi", "hello", "hey", "greetings"}
        if len(words) <= 2 and any(word.lower() in greeting_words for word in words):
            logging.info("Greeting detected, using default session name")
            return "New Chat"
        
        # Generate a name from the first few words (max 5 words, max 30 characters)
        if len(words) > 5:
            words = words[:5]
        
        session_name = " ".join(words)
        if len(session_name) > 30:
            session_name = session_name[:27] + "..."
            
        logging.info(f"Generated session name: '{session_name}' from input")
        return session_name
    except Exception as e:
        logging.error(f"Error generating session name: {str(e)}")
        return "New Chat"


def create_index(dimension):
    """Create a CPU FAISS index"""
    try:
        return faiss.IndexFlatL2(dimension)
    except Exception as e:
        logging.error(f"Error creating FAISS index: {e}")
        return None


def test_vector_store():
    """Test if vector store is working properly"""
    try:
        logging.info("Testing vector store...")
        
        # Initialize embeddings
        embeddings = GoogleGenerativeAIEmbeddings(
            google_api_key=GOOGLE_API_KEY,
            model="models/embedding-004",  # Changed from text-embedding-004
        )
        
        # Test if vector store can be loaded
        try:
            # Removed allow_dangerous_deserialization
            vectorstore = FAISS.load_local(folder_path="Faiss", embeddings=embeddings)
            logging.info("Vector store loaded successfully")
            
            # Test a simple query
            try:
                results = vectorstore.similarity_search("test query", k=1)
                logging.info(f"Search successful, found {len(results)} results")
                return True
            except Exception as search_error:
                logging.error(f"Search error: {str(search_error)}")
                return False
                
        except Exception as e:
            logging.error(f"Error loading vector store: {str(e)}")
            return False
            
    except Exception as e:
        logging.error(f"Error testing vector store: {str(e)}")
        return False


# Add this function early in your file
def should_use_dummy_data(query):
    """Determine if a query should use dummy data instead of Gemini"""
    query_lower = query.lower()

    # First check for specific case numbers or traffic violation IDs
    if re.search(r"([A-Z]+-\d{4}-\d{3})", query) or re.search(
        r"(TV-\d{4}-\d{3})", query
    ):
        return True

    # Check for vehicle registration numbers
    if re.search(r"([A-Z]{2}-\d{2}-[A-Z]{2}-\d{4})", query):
        return True

    # Specific phrases that should ALWAYS use dummy data
    exact_phrases = [
        "court vacancy",
        "vacancies",
        "judge shortage",
        "judicial vacancy",
        "case status",
        "next hearing",
        "pending case",
        "traffic challan",
        "violation",
        "fine amount",
        "fast track court",
        "pocso court",
        "women's safety court",
        "live court",
        "live streaming",
        "court proceedings",
    ]

    # Check for exact phrases
    for phrase in exact_phrases:
        if phrase in query_lower:
            return True

    # All other queries should use Gemini, especially queries about laws, articles, constitution, etc.
    return False


def is_greeting(query):
    """Check if message is just a greeting"""
    greetings = [
        "hi",
        "hello",
        "hey",
        "greetings",
        "good morning",
        "good afternoon",
        "good evening",
        "howdy",
        "what's up",
        "namaste",
        "hola",
        "hi there",
        "hello there",
        "help",
        "can you help",
        "what can you do",
        "what do you do"
    ]

    # Clean and normalize the query
    query_clean = query.lower().strip().rstrip("!.,?")
    words = query_clean.split()
    
    # Check for exact matches
    if query_clean in greetings:
        return True
        
    # Check for greeting phrases (2-3 words)
    if len(words) <= 3:
        # Check if any word is a greeting
        if any(g in query_clean for g in greetings):
            return True
        # Check for help/capability questions
        if any(phrase in query_clean for phrase in ["what can", "how can", "can you"]):
            return True
            
    return False


def process_with_gemini(prompt):
    """Process query with Gemini AI for legal advice"""
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        st.session_state.messages.append({"role": "user", "content": prompt})

        response_placeholder = st.empty()
        response_placeholder.markdown("Thinking...")

        # Get AI response
        try:
            response = user_input(prompt)

            # Save response
            st.session_state.messages.append({"role": "assistant", "content": response})
            response_placeholder.markdown(response)
            save_chat(prompt, response)
        except Exception as e:
            error_msg = f"Error generating response: {str(e)}"
            logging.error(error_msg)
            response_placeholder.markdown(
                "I apologize, but I encountered an error while processing your request. Please try again."
            )


# Add the save_chat function (after message_to_docstore and before chat_interface)
def save_chat(prompt, response):
    """Save chat messages to database"""
    try:
        if st.session_state.username:
            # Add user message
            add_chat(username=st.session_state.username, role="user", content=prompt)
            # Add assistant response
            add_chat(
                username=st.session_state.username, role="assistant", content=response
            )
            logging.info(f"Saved chat messages to database for user {st.session_state.username}")
            
            # Save the whole session for chat history
            try:
                if "messages" in st.session_state:
                    session_name = st.session_state.get("current_session", "New Chat")
                    # Generate session name from first user message if not provided or is still default
                    if not session_name or session_name == "New Chat":
                        session_name = generate_session_name(prompt)
                        st.session_state.current_session = session_name
                    
                    logging.info(f"Saving session '{session_name}' with {len(st.session_state.messages)} messages")
                    result = save_session(
                        username=st.session_state.username,
                        session_name=session_name,
                        session_data=st.session_state.messages
                    )
                    
                    if result:
                        logging.info(f"Session saved successfully: {session_name}")
                    else:
                        logging.warning(f"Failed to save session: {session_name}")
            except Exception as session_error:
                logging.error(f"Error saving session: {str(session_error)}")
        else:
            logging.warning("Chat not saved: No user logged in")
    except Exception as e:
        logging.error(f"Error saving chat: {str(e)}")
        # Continue execution even if save fails


if __name__ == "__main__":
    main()

# Add custom CSS to control font size
st.markdown(
    """
    <style>
    .stMarkdown {
        font-size: 14px;
    }
    </style>
""",
    unsafe_allow_html=True,
)

def get_context(query):
    """Get relevant context from vector store"""
    global vectorstore
    
    if vectorstore is None:
        logging.warning("Vector store not initialized")
        return None
        
    try:
        combined_results = vectorstore.similarity_search(query, k=3)
        logging.info(f"Total context documents: {len(combined_results)}")
        return combined_results
    except Exception as e:
        logging.warning(f"No context documents found: {str(e)}")
        return None
