import os
import tempfile
import uuid
from typing import List, Dict, Any, Optional
import pypdf
import docx2txt
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from app.core.config import settings

class FileProcessor:
    """Utility class for processing different file types and creating vector stores."""
    
    def __init__(self):
        """Initialize the file processor with default text splitter and embeddings."""
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        # Using a lightweight embedding model that doesn't require GPU
        self.embeddings = HuggingFaceEmbeddings(
            model_name="Qwen3-Embedding-8B",
            model_kwargs={'device': 'gpu' if settings.use_gpu else 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
    def is_valid_extension(self, filename: str) -> bool:
        """Check if the file has an allowed extension."""
        return filename.split('.')[-1].lower() in settings.allowed_extensions
    
    def extract_text_from_file(self, file_path: str) -> str:
        """Extract text from different file types."""
        file_extension = file_path.split('.')[-1].lower()
        
        if file_extension == 'pdf':
            return self._extract_text_from_pdf(file_path)
        elif file_extension == 'docx':
            return self._extract_text_from_docx(file_path)
        elif file_extension == 'txt':
            return self._extract_text_from_txt(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")
    
    def _extract_text_from_pdf(self, file_path: str) -> str:
        """Extract text from PDF files."""
        text = ""
        with open(file_path, 'rb') as file:
            pdf_reader = pypdf.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
        return text
    
    def _extract_text_from_docx(self, file_path: str) -> str:
        """Extract text from DOCX files."""
        return docx2txt.process(file_path)
    
    def _extract_text_from_txt(self, file_path: str) -> str:
        """Extract text from TXT files."""
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    
    def process_file(self, file_path: str, file_id: Optional[str] = None) -> str:
        """Process a file and add it to the vector store."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if not self.is_valid_extension(file_path):
            raise ValueError(f"Invalid file extension for file: {file_path}")
        
        # Extract text from the file
        text = self.extract_text_from_file(file_path)
        
        # Split text into chunks
        chunks = self.text_splitter.split_text(text)
        
        # Generate a unique ID for this document if not provided
        if file_id is None:
            file_id = str(uuid.uuid4())
        
        # Create metadata for each chunk
        metadatas = [{"source": file_path, "file_id": file_id} for _ in chunks]
        
        # Create or update the vector store
        vector_store = Chroma.from_texts(
            texts=chunks,
            embedding=self.embeddings,
            metadatas=metadatas,
            persist_directory=settings.vector_store_dir
        )
        
        # Persist the vector store
        vector_store.persist()
        
        return file_id
    
    def save_uploaded_file(self, file_content: bytes, filename: str) -> str:
        """Save an uploaded file to the data directory and return the file path."""
        # Generate a unique filename to avoid collisions
        unique_filename = f"{uuid.uuid4()}_{filename}"
        file_path = os.path.join(settings.data_dir, unique_filename)
        
        # Save the file
        with open(file_path, 'wb') as f:
            f.write(file_content)
        
        return file_path
    
    def get_vector_store(self):
        """Get the vector store for querying."""
        return Chroma(
            persist_directory=settings.vector_store_dir,
            embedding_function=self.embeddings
        )