import logging

# Configure logging at the top of the file
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ingest.log'),
        logging.StreamHandler()
    ]
)

# Create logger instance
logger = logging.getLogger(__name__)

from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from tqdm import tqdm
import shutil
from datetime import datetime
import json
from typing import List, Dict
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader

# Load environment variables
load_dotenv()

# Get API key from environment
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in environment variables")

# Configure Google Generative AI
genai.configure(api_key=GOOGLE_API_KEY)

class DocumentProcessor:
    def __init__(self, chunk_size=1000, chunk_overlap=200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        try:
            self.embeddings = GoogleGenerativeAIEmbeddings(
                model="models/text-embedding-004",
                google_api_key=GOOGLE_API_KEY
            )
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap
            )
        except Exception as e:
            logger.error(f"Error initializing DocumentProcessor: {e}")
            raise

    def process_pdf(self, pdf_path: str) -> List[str]:
        """Process a single PDF file"""
        try:
            logger.info(f"Processing PDF: {pdf_path}")
            reader = PdfReader(pdf_path)
            text = ""
            for page in reader.pages:
                text += page.extract_text()
            
            # Extract metadata
            metadata = {
                'source': pdf_path,
                'type': 'case_study' if 'cases' in pdf_path.lower() else 'legal_document'
            }
            
            chunks = self.text_splitter.split_text(text)
            return chunks, metadata
        except Exception as e:
            logger.error(f"Error processing PDF {pdf_path}: {e}")
            return [], {}

    def process_directory(self, directory: str) -> List[Dict]:
        """Process all PDFs in a directory"""
        all_chunks = []
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith('.pdf'):
                    file_path = os.path.join(root, file)
                    chunks, metadata = self.process_pdf(file_path)
                    if chunks:
                        all_chunks.extend([{
                            'text': chunk,
                            'metadata': {
                                **metadata,
                                'chunk_index': i
                            }
                        } for i, chunk in enumerate(chunks)])
        return all_chunks

def verify_pdf_processing(pdf_path):
    """Verify if a PDF was processed successfully"""
    try:
        reader = PdfReader(pdf_path)
        num_pages = len(reader.pages)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        return True, num_pages, len(text)
    except Exception as e:
        return False, 0, 0

def get_pdf_text(pdf_docs):
    """Extract text from PDFs with verification"""
    text = ""
    processed_files = []
    failed_files = []
    
    logging.info(f"Starting to process {len(pdf_docs)} PDF documents")
    
    with tqdm(total=len(pdf_docs), desc="Processing PDFs") as pbar:
        for pdf in pdf_docs:
            try:
                success, num_pages, text_length = verify_pdf_processing(pdf)
                if success:
                    pdf_reader = PdfReader(pdf)
                    for page in pdf_reader.pages:
                        text += page.extract_text()
                    processed_files.append({
                        'file': pdf,
                        'pages': num_pages,
                        'text_length': text_length
                    })
                    logging.info(f"Successfully processed {pdf} - {num_pages} pages")
                else:
                    failed_files.append(pdf)
                    logging.error(f"Failed to process {pdf}")
                pbar.update(1)
            except Exception as e:
                failed_files.append(pdf)
                logging.error(f"Error processing {pdf}: {str(e)}")
                pbar.update(1)
    
    # Log processing summary
    logging.info(f"\nProcessing Summary:")
    logging.info(f"Total PDFs: {len(pdf_docs)}")
    logging.info(f"Successfully processed: {len(processed_files)}")
    logging.info(f"Failed: {len(failed_files)}")
    
    return text, processed_files, failed_files

def verify_vector_store(vector_store, processed_files):
    """Verify vector store creation"""
    try:
        # Get total vectors in store
        total_vectors = len(vector_store.index_to_docstore_id)
        
        # Calculate expected chunks based on processed text
        total_text = sum(file['text_length'] for file in processed_files)
        expected_chunks = total_text // 1000  # Approximate chunk size
        
        logging.info(f"\nVector Store Verification:")
        logging.info(f"Total vectors: {total_vectors}")
        logging.info(f"Expected chunks (approximate): {expected_chunks}")
        
        # Perform test queries
        test_results = []
        for file in processed_files:
            # Extract a sample text from the file for testing
            with open(file['file'], 'rb') as f:
                pdf = PdfReader(f)
                sample_text = pdf.pages[0].extract_text()[:100]  # First 100 chars
                
            # Search for this text
            results = vector_store.similarity_search(sample_text, k=1)
            test_results.append(len(results) > 0)
        
        success_rate = sum(test_results) / len(test_results) * 100
        logging.info(f"Query test success rate: {success_rate:.2f}%")
        
        return total_vectors, success_rate
        
    except Exception as e:
        logging.error(f"Error verifying vector store: {str(e)}")
        return 0, 0

def create_vector_store():
    """Create and save the vector store"""
    try:
        processor = DocumentProcessor()
        
        # Process both general legal documents and case studies
        legal_docs = processor.process_directory("dataset/legal_documents")
        case_studies = processor.process_directory("dataset/case_studies")
        
        # Combine all documents
        all_docs = legal_docs + case_studies
        
        if not all_docs:
            logger.warning("No documents were processed successfully")
            return False
            
        # Create texts and metadatas lists
        texts = [doc['text'] for doc in all_docs]
        metadatas = [doc['metadata'] for doc in all_docs]
        
        # Create and save vector store
        vectorstore = FAISS.from_texts(
            texts=texts,
            embedding=processor.embeddings,
            metadatas=metadatas
        )
        
        # Save the vector store
        vectorstore.save_local("Faiss")
        
        # Save metadata separately for quick access
        with open("Faiss/metadata.json", "w") as f:
            json.dump(metadatas, f)
        
        logger.info(f"Vector store created with {len(texts)} chunks")
        return True
        
    except Exception as e:
        logger.error(f"Error creating vector store: {e}")
        return False

def load_pdfs(directory: str = "dataset") -> List[Dict]:
    """Load PDFs from directory and convert to documents"""
    documents = []
    
    try:
        # Walk through directory
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith('.pdf'):
                    pdf_path = os.path.join(root, file)
                    try:
                        logger.info(f"Processing {pdf_path}")
                        loader = PyPDFLoader(pdf_path)
                        pdf_docs = loader.load()
                        
                        # Add metadata
                        for doc in pdf_docs:
                            doc.metadata.update({
                                "source": file,
                                "type": "case_study" if "case" in file.lower() else "legal_principle"
                            })
                        documents.extend(pdf_docs)
                        
                    except Exception as e:
                        logger.error(f"Error processing {file}: {str(e)}")
                        continue
                        
        logger.info(f"Successfully loaded {len(documents)} documents")
        return documents
        
    except Exception as e:
        logger.error(f"Error loading PDFs: {str(e)}")
        return []

def chunk_documents(documents: List[Dict]) -> List[Dict]:
    """Split documents into chunks"""
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        
        chunks = text_splitter.split_documents(documents)
        logger.info(f"Created {len(chunks)} chunks from {len(documents)} documents")
        return chunks
        
    except Exception as e:
        logger.error(f"Error chunking documents: {str(e)}")
        return []

def create_vectorstore(chunks: List[Dict]) -> bool:
    """Create and save vector store"""
    try:
        # Initialize embeddings
        embeddings = GoogleGenerativeAIEmbeddings(
            google_api_key=GOOGLE_API_KEY,
            model="models/embedding-001"
        )
        
        # Create vector store
        vectorstore = FAISS.from_documents(
            documents=chunks,
            embedding=embeddings
        )
        
        # Save vector store
        vectorstore.save_local("Faiss")
        logger.info("Successfully created and saved vector store")
        return True
        
    except Exception as e:
        logger.error(f"Error creating vector store: {str(e)}")
        return False

def process_all_documents():
    """Process all documents and create vector store"""
    try:
        # Ensure Faiss directory exists
        os.makedirs("Faiss", exist_ok=True)
        
        # Load PDFs
        documents = load_pdfs()
        if not documents:
            logger.error("No documents loaded")
            return False
            
        # Chunk documents
        chunks = chunk_documents(documents)
        if not chunks:
            logger.error("No chunks created")
            return False
            
        # Create vector store
        success = create_vectorstore(chunks)
        return success
        
    except Exception as e:
        logger.error(f"Error in document processing: {str(e)}")
        return False

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
        
        if not pdf_docs:
            logging.error("No PDF documents found in dataset/case_studies or dataset/legal_documents directories")
            logging.info("Please ensure PDFs are placed in the correct directories:")
            logging.info("- dataset/case_studies/: for case law documents")
            logging.info("- dataset/legal_documents/: for general legal documents")
            return
        
        logging.info(f"Found total {len(pdf_docs)} PDF documents")
        
        # Process PDFs
        raw_text, processed_files, failed_files = get_pdf_text(pdf_docs)
        
        # Create text chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        text_chunks = text_splitter.split_text(raw_text)
        
        logging.info(f"\nText Chunking:")
        logging.info(f"Total chunks created: {len(text_chunks)}")
        logging.info(f"Average chunk size: {sum(len(chunk) for chunk in text_chunks) / len(text_chunks):.2f} characters")
        
        # Create vector store
        success = create_vector_store()
        
        if success:
            logging.info("Vector store created successfully!")
        else:
            logging.error("Failed to create vector store")
        
        # Final report
        print("\nProcessing Complete!")
        print(f"Total PDFs processed: {len(processed_files)}")
        print(f"Failed PDFs: {len(failed_files)}")
        
        if failed_files:
            print("\nFailed files:")
            for file in failed_files:
                print(f"- {file}")
        
    except Exception as e:
        logging.error(f"Error in main process: {str(e)}")
        raise

if __name__ == "__main__":
    # Create directories if they don't exist
    os.makedirs(os.path.join("dataset", "case_studies"), exist_ok=True)
    os.makedirs(os.path.join("dataset", "legal_documents"), exist_ok=True)
    
    success = process_all_documents()
    if success:
        print("\n✅ Successfully processed documents and created vector store!")
        print("You can now run the application with: python landing.py")
    else:
        print("\n❌ Failed to process documents.")
        print("Please check the logs for more information.")
