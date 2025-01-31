# AI-Vakeel-AI-Legal-Assistant

AI-Vakeel is an intelligent legal assistant that provides quick and accurate legal advice based on Indian laws. Built with Streamlit and powered by Google's Gemini AI, it offers an accessible platform for legal guidance.

## Features

- 🤖 AI-powered legal consultation
- 🔒 Secure user authentication system
- 💬 Chat-based interface with conversation history
- 📱 Responsive and user-friendly design
- 🔍 Context-aware legal advice
- 💾 Session management capabilities

## Tech Stack

- Python 3.8+
- Streamlit
- Google Gemini AI
- LangChain
- FAISS for vector storage
- SQLite for user management
- PyPDF2 for document processing

## Installation

1. Clone the repository:

```bash
git clone https://github.com/vishaldhavali/AI-Vakeel.git
cd AI-Vakeel
```

2. Create and activate virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Set up environment variables:
   Create a `.env` file and add:

```bash
GOOGLE_API_KEY=your_google_api_key
```

5. Run the application:

```bash
streamlit run app.py
```

## Usage

1. Sign up/Login to access the system
2. Type your legal query in the chat interface
3. Receive AI-generated legal advice based on Indian laws
4. View and manage your chat history
5. Create and switch between different consultation sessions

## Project Structure

```
AI-Vakeel/
├── app.py            # Main application file
├── db.py             # Database operations
├── ingest.py         # PDF processing and vectorization
├── dataset/          # Legal document storage
├── sessions/         # User session storage
└── requirements.txt  # Project dependencies
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author

**Vishal Dhavali**

- GitHub: [@vishaldhavali](https://github.com/vishaldhavali)

## Acknowledgments

- Thanks to Google Gemini AI for powering the legal assistance
- Streamlit for the wonderful web framework
- LangChain for AI integration capabilities
