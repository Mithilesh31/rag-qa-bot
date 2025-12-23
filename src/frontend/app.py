import streamlit as st
import os
import tempfile
import sys
from datetime import datetime
import time

# Add backend to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))

# Import SimpleRAGSystem with error handling
try:
    from simple_rag import SimpleRAGSystem
    RAG_AVAILABLE = True
except ImportError as e:
    st.error(f"❌ Failed to import SimpleRAGSystem: {e}")
    st.error("Make sure backend/simple_rag.py exists with SimpleRAGSystem class")
    RAG_AVAILABLE = False
    SimpleRAGSystem = None

# Page configuration
st.set_page_config(
    page_title="RAG QA Bot 🤖",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    /* Main title */
    .main-title {
        text-align: center;
        color: #1E3A8A;
        font-size: 2.8rem;
        margin-bottom: 0.5rem;
        font-weight: bold;
    }
    
    /* Subtitle */
    .subtitle {
        text-align: center;
        color: #6B7280;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    
    /* Answer box */
    .answer-box {
        background: linear-gradient(135deg, #F0F9FF 0%, #E0F2FE 100%);
        padding: 25px;
        border-radius: 12px;
        border-left: 6px solid #3B82F6;
        margin: 20px 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
    }
    
    /* Source box */
    .source-box {
        background-color: #F8FAFC;
        padding: 18px;
        border-radius: 10px;
        margin: 12px 0;
        border: 1px solid #E2E8F0;
        transition: all 0.3s ease;
    }
    
    .source-box:hover {
        background-color: #F1F5F9;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    
    /* Buttons */
    .stButton > button {
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3);
    }
    
    /* Metrics */
    .metric-box {
        background: white;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    /* History item */
    .history-item {
        background: #F8FAFC;
        padding: 15px;
        border-radius: 8px;
        margin: 8px 0;
        border-left: 4px solid #3B82F6;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
def init_session_state():
    if 'rag' not in st.session_state:
        if RAG_AVAILABLE:
            try:
                st.session_state.rag = SimpleRAGSystem()
                st.session_state.rag_initialized = True
            except Exception as e:
                st.error(f"Failed to initialize RAG system: {e}")
                st.session_state.rag = None
                st.session_state.rag_initialized = False
        else:
            st.session_state.rag = None
            st.session_state.rag_initialized = False
    
    if 'processed_files' not in st.session_state:
        st.session_state.processed_files = []
    
    if 'qa_history' not in st.session_state:
        st.session_state.qa_history = []
    
    if 'processing_complete' not in st.session_state:
        st.session_state.processing_complete = False
    
    if 'current_answer' not in st.session_state:
        st.session_state.current_answer = None
    
    if 'current_sources' not in st.session_state:
        st.session_state.current_sources = []
    
    if 'should_process' not in st.session_state:
        st.session_state.should_process = False
    
    if 'should_answer' not in st.session_state:
        st.session_state.should_answer = False
    
    if 'uploaded_file_path' not in st.session_state:
        st.session_state.uploaded_file_path = None
    
    if 'current_question' not in st.session_state:
        st.session_state.current_question = ""

# Main function
def main():
    # Header
    st.markdown('<h1 class="main-title">🤖 RAG QA Bot</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Upload PDFs • Ask Questions • Get Answers</p>', unsafe_allow_html=True)
    
    # Check if RAG is available
    if not st.session_state.rag_initialized:
        st.error("""
        ## ⚠️ Backend Not Available
        
        The RAG backend is not properly configured. Please ensure:
        
        1. `backend/simple_rag.py` exists with `SimpleRAGSystem` class
        2. The class has at least these methods:
           - `process_pdf(file_path)`
           - `query(question)`
        
        See the example below for a minimal implementation:
        """)
        
        with st.expander("📝 Minimal simple_rag.py example"):
            st.code("""
import os
from dotenv import load_dotenv

class SimpleRAGSystem:
    def __init__(self):
        load_dotenv()
        self.documents = []
        print("SimpleRAGSystem initialized")
    
    def process_pdf(self, pdf_path):
        print(f"Processing PDF: {pdf_path}")
        # Simple mock processing
        self.documents.append({"path": pdf_path, "content": "Sample content"})
        return True
    
    def query(self, question):
        return f"Answer to: '{question}'\\n\\nThis is a sample answer from the RAG system."
    
    def get_stats(self):
        return {
            "documents_loaded": len(self.documents),
            "chunk_size": "512"
        }
            """, language="python")
        return
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 📁 Document Upload")
        
        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type=["pdf"],
            help="Upload any PDF document to ask questions about",
            key="file_uploader"
        )
        
        if uploaded_file is not None:
            # Show file info
            file_details = {
                "Filename": uploaded_file.name,
                "File size": f"{uploaded_file.size / 1024:.1f} KB"
            }
            st.json(file_details)
            
            # Save to temp file immediately
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                st.session_state.uploaded_file_path = tmp_file.name
            
            # Process button
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🚀 Process", type="primary", use_container_width=True, key="process_btn"):
                    st.session_state.should_process = True
                    st.rerun()
            
            with col2:
                if st.button("🗑️ Clear", use_container_width=True, key="sidebar_clear_btn"):
                    # Clean up temp file
                    if st.session_state.uploaded_file_path and os.path.exists(st.session_state.uploaded_file_path):
                        try:
                            os.unlink(st.session_state.uploaded_file_path)
                        except:
                            pass
                    
                    # Reset states
                    st.session_state.processed_files = []
                    st.session_state.qa_history = []
                    st.session_state.processing_complete = False
                    st.session_state.current_answer = None
                    st.session_state.current_sources = []
                    st.session_state.should_process = False
                    st.session_state.should_answer = False
                    st.session_state.uploaded_file_path = None
                    st.session_state.current_question = ""
                    st.rerun()
        
        # Handle processing if triggered
        if st.session_state.should_process and st.session_state.uploaded_file_path:
            with st.spinner("Processing PDF..."):
                try:
                    # Use the correct method name
                    if hasattr(st.session_state.rag, 'process_pdf'):
                        success = st.session_state.rag.process_pdf(st.session_state.uploaded_file_path)
                    elif hasattr(st.session_state.rag, 'process_document'):
                        success = st.session_state.rag.process_document(st.session_state.uploaded_file_path)
                    else:
                        st.error("RAG system missing process method!")
                        success = False
                    
                    if success:
                        st.success("✅ Document processed successfully!")
                        if uploaded_file and uploaded_file.name not in st.session_state.processed_files:
                            st.session_state.processed_files.append(uploaded_file.name)
                        st.session_state.processing_complete = True
                        
                        # Show stats if available
                        if hasattr(st.session_state.rag, 'get_stats'):
                            try:
                                stats = st.session_state.rag.get_stats()
                                st.markdown("### 📊 Statistics")
                                if "documents_loaded" in stats:
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        st.metric("Chunks", stats.get("documents_loaded", 0))
                                    with col2:
                                        chunk_size = stats.get("chunk_size", "N/A")
                                        st.metric("Chunk Size", f"{chunk_size}")
                            except Exception as e:
                                st.warning(f"Could not get stats: {e}")
                    else:
                        st.error("❌ Failed to process PDF")
                    
                    # Reset the flag
                    st.session_state.should_process = False
                    
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    st.session_state.should_process = False
        
        # Status section
        st.markdown("---")
        st.markdown("### 📊 Status")
        
        if st.session_state.processed_files:
            for file in st.session_state.processed_files:
                st.success(f"✅ **{file}**")
            st.info(f"**Files processed:** {len(st.session_state.processed_files)}")
        else:
            st.warning("⚠️ No documents processed")
        
        # API Key check
        st.markdown("---")
        st.markdown("### 🔑 API Status")
        api_key = os.getenv("COHERE_API_KEY", "")
        if api_key and api_key != "your_cohere_api_key_here":
            st.success("✅ API Key: Configured")
        else:
            st.warning("⚠️ API Key: Not configured")
            st.info("Set COHERE_API_KEY in .env file")
    
    # Main content area
    tab1, tab2, tab3 = st.tabs(["💬 Ask Questions", "📚 History", "ℹ️ About"])
    
    with tab1:
        st.markdown("### 💭 Ask a Question")
        
        # Question input - bind to session state
        question = st.text_area(
            "Enter your question about the document:",
            placeholder="E.g., What is the main topic? What are the key findings? Summarize the document...",
            height=120,
            key="question_input",
            value=st.session_state.current_question
        )
        
        # Update session state with current question
        if question != st.session_state.current_question:
            st.session_state.current_question = question
        
        # Ask button
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            if st.button("🔍 Get Answer", type="primary", use_container_width=True, key="ask_btn"):
                if question:
                    st.session_state.should_answer = True
                    st.rerun()
                else:
                    st.warning("Please enter a question first!")
        
        with col2:
            if st.button("🗑️ Clear", use_container_width=True, key="question_clear_btn"):
                st.session_state.current_question = ""
                st.session_state.current_answer = None
                st.session_state.current_sources = []
                st.rerun()
        
        # Handle answering if triggered
        if st.session_state.should_answer and st.session_state.current_question:
            if not st.session_state.processing_complete:
                st.warning("⚠️ Please upload and process a PDF first!")
                st.session_state.should_answer = False
            else:
                with st.spinner("🔍 Searching for relevant information..."):
                    try:
                        # Get answer
                        if hasattr(st.session_state.rag, 'answer_question'):
                            result = st.session_state.rag.answer_question(st.session_state.current_question)
                            answer = result.get("answer", "No answer provided")
                            sources = result.get("sources", [])
                        elif hasattr(st.session_state.rag, 'query'):
                            answer = st.session_state.rag.query(st.session_state.current_question)
                            sources = []
                        else:
                            st.error("RAG system missing query method!")
                            answer = "System error: No query method found"
                            sources = []
                        
                        # Store in session state
                        st.session_state.current_answer = answer
                        st.session_state.current_sources = sources
                        
                        # Add to history
                        history_entry = {
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "question": st.session_state.current_question,
                            "answer": answer,
                            "sources_count": len(sources)
                        }
                        st.session_state.qa_history.append(history_entry)
                        
                        # Reset the flag
                        st.session_state.should_answer = False
                        
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
                        st.session_state.should_answer = False
        
        # Display current answer if available
        if st.session_state.current_answer:
            st.markdown("### 🤖 Answer")
            st.markdown('<div class="answer-box">', unsafe_allow_html=True)
            st.write(st.session_state.current_answer)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Display sources if available
            if st.session_state.current_sources:
                st.markdown(f"### 📚 Relevant Sources ({len(st.session_state.current_sources)} found)")
                
                for i, source in enumerate(st.session_state.current_sources):
                    with st.expander(f"Source {i+1}"):
                        st.markdown('<div class="source-box">', unsafe_allow_html=True)
                        if isinstance(source, dict):
                            st.write(source.get("text", "No text"))
                            if "metadata" in source:
                                st.caption(f"From: {source['metadata'].get('source', 'Unknown')}")
                        else:
                            st.write(str(source))
                        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab2:
        st.markdown("### 📚 Question History")
        
        if not st.session_state.qa_history:
            st.info("No questions asked yet. Go to 'Ask Questions' tab to start!")
        else:
            # Show history in reverse chronological order
            for i, entry in enumerate(reversed(st.session_state.qa_history)):
                with st.container():
                    st.markdown(f'<div class="history-item">', unsafe_allow_html=True)
                    
                    col1, col2 = st.columns([4, 1])
                    with col1:
                        st.markdown(f"**Q:** {entry['question']}")
                        # Show first 150 characters of answer
                        if len(entry['answer']) > 150:
                            preview = entry['answer'][:150] + "..."
                        else:
                            preview = entry['answer']
                        st.markdown(f"**A:** {preview}")
                    with col2:
                        st.caption(entry['timestamp'])
                        st.caption(f"{entry['sources_count']} sources")
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Show full answer on click
                    if st.button(f"View Details", key=f"detail_{i}_{int(time.time())}"):
                        st.write(f"**Full Answer:**")
                        st.write(entry["answer"])
                    
                    st.markdown("---")
    
    with tab3:
        st.markdown("### ℹ️ About This App")
        st.markdown("""
        #### 🎯 What is this?
        This is a **RAG (Retrieval-Augmented Generation) QA Bot** that allows you to:
        
        - **Upload** PDF documents
        - **Ask questions** about their content
        - **Get answers** with source references
        
        #### 🛠️ How it works
        1. **Document Processing**: PDFs are split into manageable chunks
        2. **Embedding Generation**: Each chunk is converted to vector embeddings
        3. **Semantic Search**: Finds most relevant chunks for your question
        4. **Answer Generation**: Creates answers based on retrieved context
        
        #### 📝 Usage Tips
        1. Click **🚀 Process** after uploading a PDF
        2. Enter your question in the text area
        3. Click **🔍 Get Answer** to get results
        4. Review source references for verification
        
        ##### 👨‍💻 Created with ❤️ for RAG QA Bot Assignment
        """)

# Run the app
if __name__ == "__main__":
    init_session_state()
    main()