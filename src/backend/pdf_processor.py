import PyPDF2
from typing import List, Tuple
import re

class SimplePDFProcessor:
    def __init__(self, chunk_size=1000, chunk_overlap=200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def load_pdf(self, pdf_path: str) -> List[str]:
        """Load PDF and split into chunks"""
        try:
            print(f"📄 Loading PDF: {pdf_path}")
            
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                # Extract text from all pages
                full_text = ""
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    text = page.extract_text()
                    full_text += text + "\n\n"
                
                # Clean text
                full_text = re.sub(r'\s+', ' ', full_text).strip()
                
                # Simple chunking by sentences
                sentences = re.split(r'(?<=[.!?])\s+', full_text)
                
                chunks = []
                current_chunk = ""
                
                for sentence in sentences:
                    if len(current_chunk) + len(sentence) < self.chunk_size:
                        current_chunk += sentence + " "
                    else:
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                        current_chunk = sentence + " "
                
                if current_chunk:
                    chunks.append(current_chunk.strip())
                
                print(f"✅ Created {len(chunks)} chunks")
                return chunks
                
        except Exception as e:
            print(f"❌ Error loading PDF: {e}")
            return []
    
    def process_pdf(self, pdf_path: str) -> Tuple[List[str], bool]:
        """Process PDF and return chunks"""
        chunks = self.load_pdf(pdf_path)
        success = len(chunks) > 0
        return chunks, success