import streamlit as st
import os
import tempfile
from typing import List, Dict, Any
from pathlib import Path
import logging

# LangChain imports
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.schema import Document

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGChatbot:
    def __init__(self):
        self.embeddings = None
        self.vectorstore = None
        self.qa_chain = None
        self.documents = []
        self.setup_openai()
    
    def setup_openai(self):
        """Initialize OpenAI components"""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            st.error("OpenAI API key not found. Please set OPENAI_API_KEY environment variable.")
            st.stop()
        
        # Initialize embeddings
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=api_key
        )
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.7,
            openai_api_key=api_key
        )
    
    def load_documents(self, uploaded_files) -> List[Document]:
        """Load and process uploaded documents"""
        documents = []
        
        for uploaded_file in uploaded_files:
            # Create temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name
            
            try:
                # Load document based on file type
                if uploaded_file.name.lower().endswith('.pdf'):
                    loader = PyPDFLoader(tmp_file_path)
                elif uploaded_file.name.lower().endswith('.txt'):
                    loader = TextLoader(tmp_file_path, encoding='utf-8')
                else:
                    st.warning(f"Unsupported file type: {uploaded_file.name}")
                    continue
                
                docs = loader.load()
                
                # Add metadata
                for doc in docs:
                    doc.metadata['source'] = uploaded_file.name
                    if 'page' not in doc.metadata:
                        doc.metadata['page'] = 'N/A'
                
                documents.extend(docs)
                logger.info(f"Loaded {len(docs)} pages from {uploaded_file.name}")
                
            except Exception as e:
                st.error(f"Error loading {uploaded_file.name}: {str(e)}")
                logger.error(f"Error loading {uploaded_file.name}: {str(e)}")
            
            finally:
                # Clean up temporary file
                if os.path.exists(tmp_file_path):
                    os.unlink(tmp_file_path)
        
        return documents
    
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks"""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        chunks = text_splitter.split_documents(documents)
        logger.info(f"Created {len(chunks)} chunks from {len(documents)} documents")
        return chunks
    
    def create_vectorstore(self, chunks: List[Document]):
        """Create FAISS vector store from document chunks"""
        try:
            if chunks:
                self.vectorstore = FAISS.from_documents(chunks, self.embeddings)
                logger.info("Vector store created successfully")
            else:
                st.warning("No chunks available to create vector store")
        except Exception as e:
            st.error(f"Error creating vector store: {str(e)}")
            logger.error(f"Error creating vector store: {str(e)}")
    
    def setup_qa_chain(self):
        """Setup the RetrievalQA chain"""
        if self.vectorstore is None:
            st.error("Vector store not initialized")
            return
        
        try:
            retriever = self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 4}
            )
            
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=retriever,
                return_source_documents=True,
                verbose=True
            )
            logger.info("QA chain setup successfully")
        except Exception as e:
            st.error(f"Error setting up QA chain: {str(e)}")
            logger.error(f"Error setting up QA chain: {str(e)}")
    
    def process_documents(self, uploaded_files):
        """Process uploaded documents end-to-end"""
        if not uploaded_files:
            return
        
        with st.spinner("Processing documents..."):
            # Load documents
            documents = self.load_documents(uploaded_files)
            if not documents:
                st.error("No documents could be loaded")
                return
            
            # Chunk documents
            chunks = self.chunk_documents(documents)
            if not chunks:
                st.error("No chunks created from documents")
                return
            
            # Create vector store
            self.create_vectorstore(chunks)
            
            # Setup QA chain
            self.setup_qa_chain()
            
            self.documents = documents
            st.success(f"Successfully processed {len(documents)} documents into {len(chunks)} chunks")
    
    def ask_question(self, question: str) -> Dict[str, Any]:
        """Ask a question and get an answer with sources"""
        if not self.qa_chain:
            return {"error": "Please upload and process documents first"}
        
        try:
            response = self.qa_chain({"query": question})
            return {
                "answer": response["result"],
                "sources": response["source_documents"]
            }
        except Exception as e:
            logger.error(f"Error answering question: {str(e)}")
            return {"error": f"Error processing question: {str(e)}"}

def main():
    st.set_page_config(
        page_title="RAG Chatbot",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("🤖 RAG-based Document Chatbot")
    st.markdown("Upload documents (PDF or TXT) and ask questions about them!")
    
    # Initialize chatbot
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = RAGChatbot()
    
    # Sidebar for document upload
    with st.sidebar:
        st.header("📁 Document Upload")
        
        uploaded_files = st.file_uploader(
            "Choose files",
            accept_multiple_files=True,
            type=['pdf', 'txt'],
            help="Upload PDF or TXT files to create a knowledge base"
        )
        
        if st.button("Process Documents", type="primary"):
            if uploaded_files:
                st.session_state.chatbot.process_documents(uploaded_files)
            else:
                st.warning("Please upload at least one file")
        
        # Display processed documents
        if st.session_state.chatbot.documents:
            st.subheader("📚 Processed Documents")
            for doc in st.session_state.chatbot.documents:
                source = doc.metadata.get('source', 'Unknown')
                st.write(f"• {source}")
    
    # Main chat interface
    st.header("💬 Ask Questions")
    
    # Initialize chat history
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Display sources if available
            if message["role"] == "assistant" and "sources" in message:
                with st.expander("📖 Sources"):
                    for i, source in enumerate(message["sources"]):
                        st.markdown(f"**Source {i+1}:** {source.metadata.get('source', 'Unknown')}")
                        if source.metadata.get('page') != 'N/A':
                            st.markdown(f"**Page:** {source.metadata.get('page', 'N/A')}")
                        st.markdown(f"**Content:** {source.page_content[:200]}...")
                        st.markdown("---")
    
    # Chat input
    if prompt := st.chat_input("What would you like to know about your documents?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get response from chatbot
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.session_state.chatbot.ask_question(prompt)
                
                if "error" in response:
                    st.error(response["error"])
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": f"❌ {response['error']}"
                    })
                else:
                    st.markdown(response["answer"])
                    
                    # Display sources
                    if response["sources"]:
                        with st.expander("📖 Sources"):
                            for i, source in enumerate(response["sources"]):
                                st.markdown(f"**Source {i+1}:** {source.metadata.get('source', 'Unknown')}")
                                if source.metadata.get('page') != 'N/A':
                                    st.markdown(f"**Page:** {source.metadata.get('page', 'N/A')}")
                                st.markdown(f"**Content:** {source.page_content[:200]}...")
                                st.markdown("---")
                    
                    # Add assistant response to chat history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response["answer"],
                        "sources": response["sources"]
                    })
    
    # Footer
    st.markdown("---")
    st.markdown("Built with ❤️ using LangChain, OpenAI, and Streamlit")

if __name__ == "__main__":
    main()
