import cohere
import numpy as np
from typing import List, Dict, Tuple
import json
import os
from config import Config
from simple_pdf_processor import SimplePDFProcessor

class SimpleRAGSystem:
    def __init__(self):
        """Initialize the RAG system"""
        self.co = cohere.Client(Config.COHERE_API_KEY)
        self.pdf_processor = SimplePDFProcessor(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP
        )
        self.documents = []  # Store document chunks
        self.embeddings = []  # Store embeddings
        self.metadata = []  # Store metadata
    
    def process_document(self, pdf_path: str) -> bool:
        """Process a PDF document"""
        print(f"\n🔄 Processing document: {os.path.basename(pdf_path)}")
        
        # Load and chunk PDF
        chunks, success = self.pdf_processor.process_pdf(pdf_path)
        if not success:
            return False
        
        # Generate embeddings
        try:
            print("🔧 Generating embeddings...")
            response = self.co.embed(
                texts=chunks,
                model=Config.EMBEDDING_MODEL,
                input_type="search_document"
            )
            
            # Store everything
            self.documents = chunks
            self.embeddings = response.embeddings
            self.metadata = [{"source": os.path.basename(pdf_path), "chunk_id": i} 
                           for i in range(len(chunks))]
            
            print(f"✅ Successfully processed {len(chunks)} chunks")
            return True
            
        except Exception as e:
            print(f"❌ Error generating embeddings: {e}")
            return False
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0
        
        return dot_product / (norm1 * norm2)
    
    def search(self, query: str, top_k: int = Config.TOP_K) -> List[Dict]:
        """Search for relevant documents"""
        if not self.embeddings:
            print("⚠️ No documents loaded yet")
            return []
        
        try:
            # Embed the query
            query_response = self.co.embed(
                texts=[query],
                model=Config.EMBEDDING_MODEL,
                input_type="search_query"
            )
            query_embedding = query_response.embeddings[0]
            
            # Calculate similarities
            similarities = []
            for i, doc_embedding in enumerate(self.embeddings):
                similarity = self._cosine_similarity(query_embedding, doc_embedding)
                similarities.append((i, similarity))
            
            # Sort by similarity (highest first)
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            # Get top-k results
            results = []
            for i, (doc_idx, score) in enumerate(similarities[:top_k]):
                results.append({
                    "id": i + 1,
                    "text": self.documents[doc_idx],
                    "score": float(score),
                    "metadata": self.metadata[doc_idx]
                })
            
            return results
            
        except Exception as e:
            print(f"❌ Error during search: {e}")
            return []
    
    def answer_question(self, question: str) -> Dict:
        """Answer a question based on documents"""
        print(f"\n❓ Question: {question}")
        
        # 1. Search for relevant chunks
        relevant_chunks = self.search(question)
        
        if not relevant_chunks:
            return {
                "question": question,
                "answer": "I couldn't find any relevant information in the document. Please make sure a document has been processed first.",
                "sources": [],
                "status": "no_results"
            }
        
        # 2. Prepare context
        context_parts = []
        for chunk in relevant_chunks:
            context_parts.append(f"[Source {chunk['id']}, Relevance: {chunk['score']:.2%}]\n{chunk['text']}")
        
        context = "\n\n".join(context_parts)
        
        # 3. Generate answer
        try:
            print("🤖 Generating answer...")
            
            prompt = f"""Based on the following context from a document, answer the question.
            
CONTEXT:
{context}

QUESTION: {question}

INSTRUCTIONS:
1. Answer based ONLY on the context provided
2. If the context doesn't contain the answer, say "I cannot find this information in the document"
3. Be concise and accurate
4. Reference which source(s) you used

ANSWER:"""
            
            response = self.co.generate(
                model=Config.GENERATION_MODEL,
                prompt=prompt,
                max_tokens=500,
                temperature=0.3,
                stop_sequences=["\n\n"]
            )
            
            answer = response.generations[0].text.strip()
            
            return {
                "question": question,
                "answer": answer,
                "sources": relevant_chunks,
                "status": "success"
            }
            
        except Exception as e:
            return {
                "question": question,
                "answer": f"Error generating answer: {str(e)}",
                "sources": relevant_chunks,
                "status": "error"
            }
    
    def get_stats(self) -> Dict:
        """Get system statistics"""
        return {
            "documents_loaded": len(self.documents),
            "embeddings_created": len(self.embeddings),
            "chunk_size": Config.CHUNK_SIZE,
            "top_k": Config.TOP_K
        }

# Quick test function
def test_rag():
    """Test the RAG system"""
    print("🧪 Testing Simple RAG System...")
    
    # Initialize
    rag = SimpleRAGSystem()
    
    # Check API key
    if not Config.COHERE_API_KEY or Config.COHERE_API_KEY == "your_cohere_api_key_here":
        print("❌ Please set your COHERE_API_KEY in the .env file!")
        print("💡 Get a free key from: https://dashboard.cohere.com/")
        return
    
    print("✅ RAG system initialized successfully")
    print("📝 Add a PDF to the data/ folder and run the Streamlit app!")
    print("\nTo run the app: streamlit run src/frontend/app.py")

if __name__ == "__main__":
    test_rag()