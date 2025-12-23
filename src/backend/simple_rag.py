import cohere
import numpy as np
from typing import List, Dict
import os

# Local imports
from config import Config
from simple_pdf_processor import SimplePDFProcessor

class SimpleRAGSystem:
    def __init__(self):
        """Initialize the RAG system"""
        print("🚀 Initializing SimpleRAGSystem...")
        try:
            self.co = cohere.Client(Config.COHERE_API_KEY)
            print(f"✅ Cohere client initialized with key: {Config.COHERE_API_KEY[:10]}...")
        except Exception as e:
            print(f"❌ Failed to initialize Cohere client: {e}")
            raise
        
        self.pdf_processor = SimplePDFProcessor(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP
        )
        self.documents = []  # Store document chunks
        self.embeddings = []  # Store embeddings
        self.metadata = []  # Store metadata
        print("✅ SimpleRAGSystem initialized successfully!")
    
    def process_pdf(self, pdf_path: str) -> bool:
        """Process a PDF document - This is what the app expects"""
        return self.process_document(pdf_path)
    
    def process_document(self, pdf_path: str) -> bool:
        """Process a PDF document"""
        print(f"\n🔄 Processing document: {os.path.basename(pdf_path)}")
        
        # Check if file exists
        if not os.path.exists(pdf_path):
            print(f"❌ File not found: {pdf_path}")
            return False
        
        # Load and chunk PDF
        print("📄 Loading and chunking PDF...")
        chunks, success = self.pdf_processor.process_pdf(pdf_path)
        if not success:
            print("❌ Failed to process PDF")
            return False
        
        print(f"📊 Created {len(chunks)} chunks")
        
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
    
    def query(self, question: str) -> str:
        """Simple query interface - returns just the answer string"""
        print(f"\n❓ Query: {question}")
        
        result = self.answer_question(question)
        return result.get("answer", "No answer available")
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        try:
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0
            
            return dot_product / (norm1 * norm2)
        except Exception as e:
            print(f"❌ Error calculating cosine similarity: {e}")
            return 0
    
    def search(self, query: str, top_k: int = Config.TOP_K) -> List[Dict]:
        """Search for relevant documents"""
        if not self.embeddings:
            print("⚠️ No documents loaded yet")
            return []
        
        try:
            # Embed the query
            print("🔍 Embedding query...")
            query_response = self.co.embed(
                texts=[query],
                model=Config.EMBEDDING_MODEL,
                input_type="search_query"
            )
            query_embedding = query_response.embeddings[0]
            
            # Calculate similarities
            print("📊 Calculating similarities...")
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
            
            print(f"✅ Found {len(results)} relevant chunks")
            return results
            
        except Exception as e:
            print(f"❌ Error during search: {e}")
            return []
    
    def answer_question(self, question: str) -> Dict:
        """Answer a question based on documents using Cohere Chat API"""
        print(f"\n❓ Question: {question}")
        
        # Check if documents are loaded
        if not self.documents:
            return {
                "question": question,
                "answer": "⚠️ No documents have been processed yet. Please upload and process a PDF first.",
                "sources": [],
                "status": "no_documents"
            }
        
        # 1. Search for relevant chunks
        print("🔍 Searching for relevant information...")
        relevant_chunks = self.search(question)
        
        if not relevant_chunks:
            return {
                "question": question,
                "answer": "❌ I couldn't find any relevant information in the document for your question.",
                "sources": [],
                "status": "no_results"
            }
        
        # 2. Prepare context
        context_parts = []
        for chunk in relevant_chunks:
            context_parts.append(f"[Source {chunk['id']}, Relevance: {chunk['score']:.2%}]\n{chunk['text']}")
        
        context = "\n\n".join(context_parts)
        
        # 3. Generate answer using Chat API
        try:
            print(f"🤖 Generating answer with model: {Config.GENERATION_MODEL}")
            
            # Prepare the chat message with context
            chat_message = f"""You are a helpful assistant that answers questions based on the provided context.

CONTEXT FROM DOCUMENT:
{context}

INSTRUCTIONS:
1. Answer the question based ONLY on the context provided above
2. If the context doesn't contain the answer, say "I cannot find this information in the document"
3. Be concise and accurate
4. Reference which source(s) you used when applicable
5. If the question is about the document's main topic, summarize what the document is about

QUESTION: {question}

ANSWER:"""
            
            response = self.co.chat(
                message=chat_message,
                model=Config.GENERATION_MODEL,
                temperature=0.3,
                max_tokens=500
            )
            
            answer = response.text.strip()
            print(f"✅ Answer generated successfully")
            
            return {
                "question": question,
                "answer": answer,
                "sources": relevant_chunks,
                "status": "success",
                "model_used": Config.GENERATION_MODEL
            }
            
        except Exception as e:
            error_msg = f"❌ Error generating answer: {str(e)}"
            print(error_msg)
            
            # Fallback: return simple answer based on search results
            fallback_answer = "Based on the document, here's what I found:\n\n"
            for i, chunk in enumerate(relevant_chunks[:3], 1):
                fallback_answer += f"**Source {i}** (Relevance: {chunk['score']:.1%}):\n"
                fallback_answer += f"{chunk['text'][:200]}...\n\n"
            
            return {
                "question": question,
                "answer": fallback_answer,
                "sources": relevant_chunks,
                "status": "fallback"
            }
    
    def get_stats(self) -> Dict:
        """Get system statistics"""
        return {
            "documents_loaded": len(self.documents),
            "embeddings_created": len(self.embeddings),
            "chunk_size": Config.CHUNK_SIZE,
            "top_k": Config.TOP_K,
            "embedding_model": Config.EMBEDDING_MODEL,
            "generation_model": Config.GENERATION_MODEL
        }

# Quick test function
def test_rag():
    """Test the RAG system"""
    print("🧪 Testing Simple RAG System...")
    
    try:
        # Initialize
        print("1. Initializing RAG system...")
        rag = SimpleRAGSystem()
        
        # Check API key
        if not Config.COHERE_API_KEY or Config.COHERE_API_KEY == "your_cohere_api_key_here":
            print("❌ Please set your COHERE_API_KEY in the .env file!")
            print("💡 Get a free key from: https://dashboard.cohere.com/")
            return
        
        print("\n2. Testing methods...")
        
        # Test get_stats
        stats = rag.get_stats()
        print(f"📊 Initial stats: {stats}")
        
        # Test query (without document)
        test_question = "What is this document about?"
        print(f"\n3. Testing query without document: '{test_question}'")
        answer = rag.query(test_question)
        print(f"🤖 Answer: {answer}")
        
        print("\n✅ RAG system test completed!")
        print("\n📝 To use the system:")
        print("1. Run: streamlit run src/frontend/app.py")
        print("2. Upload a PDF in the web interface")
        print("3. Click 'Process' then ask questions!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_rag()