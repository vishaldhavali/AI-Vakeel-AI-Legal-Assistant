import streamlit as st
import json
from datetime import datetime, timedelta
import os
from cryptography.fernet import Fernet
import base64
from langchain_google_genai import ChatGoogleGenerativeAI

class SessionManager:
    def __init__(self):
        self.session_file = "sessions/user_sessions.json"
        self.session_duration = timedelta(days=7)
        self.key_file = "encryption_key.key"
        
        # Create sessions directory if it doesn't exist
        os.makedirs("sessions", exist_ok=True)
        
        # Initialize encryption
        self.fernet = self._initialize_encryption()
        
        # Initialize session state if not exists
        if "authentication_status" not in st.session_state:
            st.session_state.authentication_status = False
        if "username" not in st.session_state:
            st.session_state.username = None
        if "profile_image" not in st.session_state:
            st.session_state.profile_image = None
        if "messages" not in st.session_state:
            st.session_state.messages = []
        if "current_session" not in st.session_state:
            st.session_state.current_session = None

    def _initialize_encryption(self):
        """Initialize or load the encryption key"""
        try:
            if not os.path.exists(self.key_file):
                # Generate a new key
                key = Fernet.generate_key()
                with open(self.key_file, "wb") as key_file:
                    key_file.write(key)
                return Fernet(key)
            else:
                # Load existing key
                with open(self.key_file, "rb") as key_file:
                    key = key_file.read().strip()
                    # Verify key is valid
                    if not self._is_valid_key(key):
                        # Generate new key if existing one is invalid
                        key = Fernet.generate_key()
                        with open(self.key_file, "wb") as kf:
                            kf.write(key)
                    return Fernet(key)
        except Exception as e:
            print(f"Error initializing encryption: {e}")
            # Fallback: Generate new key
            key = Fernet.generate_key()
            with open(self.key_file, "wb") as key_file:
                key_file.write(key)
            return Fernet(key)
    
    def _is_valid_key(self, key):
        """Verify if the key is valid Fernet key"""
        try:
            # Key must be 32 url-safe base64-encoded bytes
            decoded = base64.urlsafe_b64decode(key)
            return len(decoded) == 32
        except:
            return False
    
    def load_sessions(self):
        if os.path.exists(self.session_file):
            try:
                with open(self.session_file, "rb") as f:
                    encrypted_data = f.read()
                    if encrypted_data:
                        decrypted_data = self.fernet.decrypt(encrypted_data)
                        return json.loads(decrypted_data)
            except Exception as e:
                print(f"Error loading sessions: {e}")
        return {}
    
    def save_sessions(self, sessions):
        try:
            encrypted_data = self.fernet.encrypt(json.dumps(sessions).encode())
            with open(self.session_file, "wb") as f:
                f.write(encrypted_data)
        except Exception as e:
            print(f"Error saving sessions: {e}")
    
    def _is_valid_stored_session(self, session_data):
        try:
            expires_at = datetime.fromisoformat(session_data["expires_at"])
            return expires_at > datetime.now() and session_data.get("is_authenticated", False)
        except:
            return False

    def _restore_session(self, username, session_data):
        st.session_state.authentication_status = True
        st.session_state.username = username
        st.session_state.profile_image = session_data.get("profile_image")
        st.session_state.messages = session_data.get("messages", [])
        st.session_state.current_session = session_data.get("current_session")

    def create_session(self, username, profile_image=None, email=None):
        sessions = self.load_sessions()
        sessions[username] = {
            "created_at": datetime.now().isoformat(),
            "expires_at": (datetime.now() + self.session_duration).isoformat(),
            "profile_image": profile_image,
            "email": email,
            "is_authenticated": True
        }
        self.save_sessions(sessions)
        
        # Update session state
        st.session_state.authentication_status = True
        st.session_state.username = username
        st.session_state.profile_image = profile_image
        st.session_state.email = email
        st.session_state.messages = []
        st.session_state.current_session = None

    def validate_session(self):
        """Validate current session"""
        if not st.session_state.authentication_status:
            return False
            
        username = st.session_state.username
        if not username:
            return False
            
        sessions = self.load_sessions()
        if username in sessions:
            session_data = sessions[username]
            if self._is_valid_stored_session(session_data):
                # Extend session
                sessions[username]["expires_at"] = (datetime.now() + self.session_duration).isoformat()
                self.save_sessions(sessions)
                return True
        
        return False

    def end_session(self):
        """End current session"""
        if st.session_state.username:
            username = st.session_state.username
            sessions = self.load_sessions()
            if username in sessions:
                del sessions[username]
                self.save_sessions(sessions)
        
        # Reset session state
        st.session_state.authentication_status = False
        st.session_state.username = None
        st.session_state.profile_image = None
        st.session_state.messages = []
        st.session_state.current_session = None 

    def get_model():
        return ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",  # Updated model
            temperature=0.7,
            google_api_key="AIzaSyDCRZOok-jCCpl-q5kCSXR9fhA3lpdflvY"
        ) 