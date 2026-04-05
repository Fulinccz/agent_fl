import os
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from logger import get_logger

logger = get_logger(__name__)

@dataclass
class DocumentChunk:
    content: str
    metadata: Dict[str, Any]
    chunk_id: str

class DocumentProcessor:
    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        separators: List[str] = None
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", "。", "！", "？", ".", " ", ""]
    
    def load_file(self, file_path: str) -> Optional[str]:
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return None
        
        try:
            ext = os.path.splitext(file_path)[1].lower()
            
            if ext == ".pdf":
                return self._load_pdf(file_path)
            elif ext in [".docx", ".doc"]:
                return self._load_docx(file_path)
            elif ext == ".txt":
                return self._load_txt(file_path)
            else:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    return f.read()
        except Exception as e:
            logger.error(f"Error loading file {file_path}: {e}")
            return None
    
    def _load_pdf(self, file_path: str) -> str:
        try:
            import PyPDF2
            text = ""
            with open(file_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    text += page.extract_text() or ""
            return text
        except ImportError:
            import subprocess
            result = subprocess.run(
                ['python', '-m', 'PyPDF2', file_path],
                capture_output=True, text=True
            )
            return result.stdout or ""
    
    def _load_docx(self, file_path: str) -> str:
        try:
            from docx import Document
            doc = Document(file_path)
            return "\n".join([para.text for para in doc.paragraphs])
        except ImportError:
            return ""
    
    def _load_txt(self, file_path: str) -> str:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()
    
    def split_text(
        self,
        text: str,
        metadata: Dict[str, Any] = None
    ) -> List[DocumentChunk]:
        if not text or not text.strip():
            return []
        
        metadata = metadata or {}
        
        chunks = []
        
        for separator in self.separators:
            if separator in text:
                sections = text.split(separator)
                
                current_chunk = ""
                for section in sections:
                    section = section.strip()
                    if not section:
                        continue
                    
                    test_chunk = current_chunk + ("\n" if current_chunk else "") + section + separator
                    
                    if len(test_chunk) > self.chunk_size:
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                        
                        if len(section) > self.chunk_size:
                            sub_chunks = self._split_long_text(section, metadata)
                            chunks.extend([c.content for c in sub_chunks])
                            current_chunk = ""
                        else:
                            current_chunk = section + separator
                    else:
                        current_chunk = test_chunk
                
                if current_chunk:
                    chunks.append(current_chunk.strip())
                
                break
        else:
            chunks = self._split_long_text(text, metadata)
            chunks = [c.content for c in chunks]
        
        result = []
        for i, chunk in enumerate(chunks):
            if chunk.strip():
                result.append(DocumentChunk(
                    content=chunk.strip(),
                    metadata={**metadata, "chunk_index": i, "total_chunks": len(chunks)},
                    chunk_id=f"chunk_{i}"
                ))
        
        logger.info(f"Split into {len(result)} chunks")
        return result
    
    def _split_long_text(self, text: str, metadata: Dict) -> List[DocumentChunk]:
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            
            if end < len(text):
                last_period = text.rfind("。", start, end)
                last_newline = text.rfind("\n", start, end)
                last_space = text.rfind(" ", start, end)
                
                split_pos = max(last_period, last_newline, last_space)
                
                if split_pos > start + self.chunk_size // 2:
                    end = split_pos + 1
            
            chunk_text = text[start:end].strip()
            
            if chunk_text:
                chunks.append(DocumentChunk(
                    content=chunk_text,
                    metadata=metadata or {},
                    chunk_id=f"chunk_{len(chunks)}"
                ))
            
            start = end - (self.chunk_overlap if end < len(text) else 0)
        
        return chunks
    
    def process_file(
        self,
        file_path: str,
        metadata: Dict[str, Any] = None
    ) -> List[DocumentChunk]:
        text = self.load_file(file_path)
        
        if not text:
            return []
        
        metadata = metadata or {}
        metadata["source"] = os.path.basename(file_path)
        metadata["file_path"] = file_path
        
        return self.split_text(text, metadata)
    
    def process_directory(
        self,
        dir_path: str,
        extensions: List[str] = None,
        metadata: Dict[str, Any] = None
    ) -> List[DocumentChunk]:
        extensions = extensions or ['.txt', '.md', '.pdf', '.docx']
        
        all_chunks = []
        
        for root, dirs, files in os.walk(dir_path):
            for file in files:
                ext = os.path.splitext(file)[1].lower()
                if ext in extensions:
                    file_path = os.path.join(root, file)
                    chunks = self.process_file(file_path, metadata)
                    all_chunks.extend(chunks)
        
        logger.info(f"Processed directory: {len(all_chunks)} total chunks")
        return all_chunks