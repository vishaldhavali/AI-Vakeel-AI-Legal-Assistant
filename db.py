import sqlite3
import logging
from datetime import datetime
import os
import json
import re

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)

def sanitized_filename(filename):
    """Create a safe filename from the input string"""
    # Remove invalid characters
    filename = re.sub(r'[<>:"/\\|?*]', '', filename)
    # Replace spaces with underscores
    filename = filename.replace(' ', '_')
    # Replace @ with _at_ for email addresses
    filename = filename.replace('@', '_at_')
    return filename

def create_db():
    conn = sqlite3.connect('app.db')
    c = conn.cursor()
    try:
        # Create users table
        c.execute('''CREATE TABLE IF NOT EXISTS users
                    (username TEXT PRIMARY KEY,
                     password TEXT NOT NULL,
                     email TEXT,
                     profile_image TEXT,
                     created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
        
        # Create chat_history table with proper schema
        c.execute('''CREATE TABLE IF NOT EXISTS chat_history
                    (id INTEGER PRIMARY KEY AUTOINCREMENT,
                     username TEXT NOT NULL,
                     role TEXT NOT NULL,
                     content TEXT NOT NULL,
                     timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                     FOREIGN KEY (username) REFERENCES users(username))''')
        
        # Feedback table
        c.execute('''
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY,
                username TEXT,
                feedback_type TEXT,
                feedback_text TEXT,
                timestamp REAL DEFAULT (strftime('%s', 'now'))
            )
        ''')
        
        # Analytics table
        c.execute('''
            CREATE TABLE IF NOT EXISTS analytics (
                id INTEGER PRIMARY KEY,
                username TEXT,
                session_duration REAL,
                message_count INTEGER,
                query_types TEXT,
                timestamp REAL DEFAULT (strftime('%s', 'now'))
            )
        ''')
        
        conn.commit()
    except Exception as e:
        logging.error(f"Error creating database: {e}")
        raise
    finally:
        conn.close()

def add_user(username, password, profile_image):
    conn = sqlite3.connect('app.db')
    c = conn.cursor()
    try:
        c.execute("INSERT INTO users (username, password, profile_image) VALUES (?, ?, ?)", (username, password, profile_image))
        conn.commit()
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()
    return True

def get_user(username, password):
    conn = sqlite3.connect('app.db')
    c = conn.cursor()
    try:
        c.execute("SELECT username, profile_image, email FROM users WHERE username=? AND password=?", 
                 (username, password))
        user = c.fetchone()
        if user:
            return {
                "username": user[0],
                "profile_image": user[1],
                "email": user[2]
            }
        return None
    finally:
        conn.close()

def update_profile_image(username, image_data):
    conn = sqlite3.connect('app.db')
    c = conn.cursor()
    try:
        c.execute("UPDATE users SET profile_image=? WHERE username=?", (image_data, username))
        conn.commit()
        return True
    except sqlite3.Error:
        return False
    finally:
        conn.close()

def update_user_email_with_verification(username, new_email, password):
    """Update user email with password verification"""
    conn = sqlite3.connect('app.db')
    c = conn.cursor()
    try:
        # Debug logging for troubleshooting
        logging.info(f"Starting email update for user: {username}")
        
        # First verify the password
        c.execute("SELECT password FROM users WHERE username=?", (username,))
        stored_password = c.fetchone()
        
        logging.info(f"Stored password exists: {stored_password is not None}")
        
        if not stored_password:
            logging.warning(f"User not found: {username}")
            return (False, "User not found")
        
        # Log password comparison (be careful not to log actual passwords)
        logging.info("Comparing passwords...")
        logging.info(f"Stored password length: {len(stored_password[0])}")
        logging.info(f"Provided password length: {len(password)}")
        
        # Compare passwords
        if stored_password[0] == password:  # Direct comparison since password is already hashed
            # Check if email already exists
            c.execute("SELECT username FROM users WHERE email=? AND username!=?", (new_email, username))
            existing_user = c.fetchone()
            
            if existing_user:
                logging.warning(f"Email already in use: {new_email}")
                return (False, "Email already in use by another account")
            
            # Update email
            try:
                c.execute("UPDATE users SET email=? WHERE username=?", (new_email, username))
                conn.commit()
                logging.info(f"Email updated successfully for user: {username}")
                return (True, "Email updated successfully")
            except sqlite3.Error as e:
                logging.error(f"Database error while updating email: {str(e)}")
                return (False, f"Database error: {str(e)}")
        else:
            logging.warning("Password verification failed")
            return (False, "Invalid password")
            
    except sqlite3.Error as e:
        logging.error(f"Error updating email: {str(e)}")
        return (False, f"Database error: {str(e)}")
    finally:
        conn.close()

def add_chat(username, role, content):
    """Add a chat message to the database"""
    conn = sqlite3.connect('app.db')
    c = conn.cursor()
    try:
        c.execute(
            "INSERT INTO chat_history (username, role, content) VALUES (?, ?, ?)",
            (username, role, content)
        )
        conn.commit()
    except Exception as e:
        logging.error(f"Error adding chat: {e}")
        raise
    finally:
        conn.close()

def get_chat_history(username):
    """Get chat history for a user"""
    conn = sqlite3.connect('app.db')
    c = conn.cursor()
    try:
        c.execute(
            "SELECT role, content FROM chat_history WHERE username=? ORDER BY timestamp ASC",
            (username,)
        )
        history = [{"role": role, "content": content} for role, content in c.fetchall()]
        return history
    except Exception as e:
        logging.error(f"Error getting chat history: {e}")
        return []
    finally:
        conn.close()

def clear_chat_history(username):
    conn = sqlite3.connect('app.db')
    c = conn.cursor()
    c.execute("DELETE FROM chat_history WHERE username=?", (username,))
    conn.commit()
    conn.close()

def update_password(username, current_password_hash, new_password_hash):
    conn = sqlite3.connect('app.db')
    c = conn.cursor()
    try:
        c.execute("SELECT password FROM users WHERE username=?", (username,))
        stored_password = c.fetchone()
        if stored_password and stored_password[0] == current_password_hash:
            c.execute("UPDATE users SET password=? WHERE username=?", 
                     (new_password_hash, username))
            conn.commit()
            return True
        return False
    finally:
        conn.close()

def get_total_messages(username):
    conn = sqlite3.connect('app.db')
    c = conn.cursor()
    try:
        c.execute("SELECT COUNT(*) FROM chat_history WHERE username=?", (username,))
        return c.fetchone()[0]
    finally:
        conn.close()

def get_account_age(username):
    conn = sqlite3.connect('app.db')
    c = conn.cursor()
    try:
        c.execute("SELECT created_at FROM users WHERE username=?", (username,))
        created_at = c.fetchone()
        if created_at:
            days = (datetime.now() - datetime.fromtimestamp(created_at[0])).days
            return days
        return 0
    finally:
        conn.close()

def delete_user_account(username, password_hash):
    """Delete user account and all associated data"""
    conn = sqlite3.connect('app.db')
    c = conn.cursor()
    try:
        # First verify the password
        c.execute("SELECT password FROM users WHERE username=?", (username,))
        stored_password = c.fetchone()
        if stored_password and stored_password[0] == password_hash:
            # Delete user data from all tables
            c.execute("DELETE FROM users WHERE username=?", (username,))
            c.execute("DELETE FROM chat_history WHERE username=?", (username,))
            c.execute("DELETE FROM feedback WHERE username=?", (username,))
            c.execute("DELETE FROM analytics WHERE username=?", (username,))
            conn.commit()
            return True
        return False
    except sqlite3.Error as e:
        logging.error(f"Error deleting user account: {str(e)}")
        return False
    finally:
        conn.close()

def list_sessions(username):
    """Get list of chat sessions for a user"""
    try:
        logging.info(f"Listing sessions for user: {username}")
        sessions = []
        sessions_dir = "sessions"
        
        # Create sessions directory if it doesn't exist
        if not os.path.exists(sessions_dir):
            logging.info(f"Creating missing sessions directory: {sessions_dir}")
            os.makedirs(sessions_dir)
            return []
            
        username_prefix = sanitized_filename(username)
        logging.info(f"Looking for session files with prefix: {username_prefix}_")
        
        session_files = os.listdir(sessions_dir)
        logging.info(f"Found {len(session_files)} total files in sessions directory")
        
        matching_files = [f for f in session_files if f.startswith(f"{username_prefix}_")]
        logging.info(f"Found {len(matching_files)} session files for user {username}")
        
        for filename in matching_files:
            try:
                file_path = os.path.join(sessions_dir, filename)
                if not os.path.isfile(file_path):
                    logging.warning(f"Skipping non-file item: {file_path}")
                    continue
                    
                logging.info(f"Reading session file: {file_path}")
                with open(file_path, 'r', encoding='utf-8') as f:
                    session_data = json.load(f)
                    
                    # Extract session details with proper fallbacks
                    session_name = session_data.get("session_name", "Unnamed Chat")
                    timestamp = session_data.get("timestamp", datetime.now().isoformat())
                    metadata = session_data.get("metadata", {})
                    message_count = metadata.get("message_count", 0)
                    
                    logging.info(f"Successfully read session: {session_name}, messages: {message_count}")
                    
                    sessions.append({
                        "filename": filename,
                        "session_name": session_name,
                        "timestamp": timestamp,
                        "message_count": message_count,
                        "last_message": metadata.get("last_message", ""),
                        "created_at": metadata.get("created_at", timestamp)
                    })
            except json.JSONDecodeError as json_err:
                logging.error(f"Invalid JSON in session file {filename}: {str(json_err)}")
                continue
            except Exception as e:
                logging.error(f"Error reading session file {filename}: {str(e)}")
                continue
                
        # Sort sessions by timestamp, newest first
        sessions.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        logging.info(f"Returning {len(sessions)} valid sessions for user {username}")
        return sessions
        
    except Exception as e:
        logging.error(f"Error listing sessions: {str(e)}")
        return []

def verify_profile_image(username):
    """Verify that profile image is stored correctly"""
    conn = sqlite3.connect('app.db')
    c = conn.cursor()
    try:
        c.execute("SELECT profile_image FROM users WHERE username=?", (username,))
        result = c.fetchone()
        if result:
            return result[0]  # Return the profile image data
        return None
    finally:
        conn.close()

def get_user_email(username):
    """Get user's current email"""
    conn = sqlite3.connect('app.db')
    c = conn.cursor()
    try:
        c.execute("SELECT email FROM users WHERE username=?", (username,))
        result = c.fetchone()
        return result[0] if result else None
    finally:
        conn.close()
