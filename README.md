# RAG-based Document Chatbot

A simple yet powerful RAG (Retrieval-Augmented Generation) chatbot that allows users to upload documents and ask questions about them using OpenAI's models.

## Features

- **Document Upload**: Support for PDF and TXT files
- **Intelligent Chunking**: Uses LangChain's RecursiveCharacterTextSplitter
- **Vector Search**: FAISS-based semantic search with OpenAI embeddings
- **Smart Q&A**: Powered by OpenAI's GPT models with source citations
- **Web Interface**: Clean Streamlit UI for easy interaction
- **Dockerized**: Ready-to-run container setup

## Architecture

```
User Upload → Document Loader → Text Splitter → OpenAI Embeddings → FAISS Vector Store
                                                                            ↓
User Question → Retriever → Context + Question → OpenAI GPT → Answer + Sources
```

## Quick Start

### Option 1: Docker (Recommended)

1. **Clone and build**:
```bash
git clone <repository>
cd rag-chatbot
docker build -t rag-chatbot .
```

2. **Run with your OpenAI API key**:
```bash
docker run -p 8501:8501 -e OPENAI_API_KEY=your_openai_api_key_here rag-chatbot
```

3. **Access the app**: Open http://localhost:8501

### Option 2: Docker Compose

1. **Create .env file**:
```bash
cp .env.example .env
# Edit .env and add your OpenAI API key
```

2. **Run with compose**:
```bash
docker-compose up -d
```

### Option 3: Local Development

1. **Install dependencies**:
```bash
pip install -r requirements.txt
```

2. **Set environment variable**:
```bash
export OPENAI_API_KEY=your_openai_api_key_here
```

3. **Run the app**:
```bash
streamlit run app.py
```

## How to Use

1. **Upload Documents**: 
   - Click "Choose files" in the sidebar
   - Select PDF or TXT files
   - Click "Process Documents"

2. **Ask Questions**:
   - Type your question in the chat input
   - Get AI-powered answers with source citations
   - View the source documents that informed the answer

## Configuration

### Environment Variables

- `OPENAI_API_KEY`: Your OpenAI API key (required)
- `OPENAI_EMBEDDING_MODEL`: Embedding model (default: text-embedding-3-small)
- `OPENAI_CHAT_MODEL`: Chat model (default: gpt-3.5-turbo)

### Customization

The RAG pipeline can be customized by modifying these parameters in `app.py`:

```python
# Text splitting
chunk_size=1000
chunk_overlap=200

# Vector search
search_kwargs={"k": 4}  # Number of chunks to retrieve

# LLM parameters
temperature=0.7
```

## System Requirements

- **Memory**: 2GB RAM minimum (4GB recommended)
- **CPU**: 1 core minimum (2 cores recommended)
- **Storage**: 1GB free space
- **Network**: Internet connection for OpenAI API calls

## API Usage

The system uses OpenAI's APIs:
- **Embeddings**: text-embedding-3-small (~$0.02/1M tokens)
- **Chat**: gpt-3.5-turbo (~$0.002/1K tokens)

## Troubleshooting

### Common Issues

1. **"OpenAI API key not found"**:
   - Ensure OPENAI_API_KEY environment variable is set
   - Check your API key has sufficient credits

2. **"Error loading document"**:
   - Verify file format (PDF/TXT only)
   - Check file isn't corrupted or password-protected

3. **"Vector store not initialized"**:
   - Upload and process documents first
   - Ensure documents were processed successfully

### Performance Tips

- **Large documents**: Split into smaller files for better performance
- **Many documents**: Process in batches to avoid memory issues
- **Slow responses**: Reduce retrieval count (k parameter)

## Architecture Details

### Components

1. **Document Loaders**: PyPDFLoader, TextLoader
2. **Text Splitter**: RecursiveCharacterTextSplitter
3. **Embeddings**: OpenAI text-embedding-3-small
4. **Vector Store**: FAISS (Facebook AI Similarity Search)
5. **LLM**: OpenAI GPT-3.5-turbo/GPT-4
6. **Framework**: LangChain for orchestration
7. **UI**: Streamlit for web interface

### Data Flow

1. User uploads documents via Streamlit UI
2. Documents are loaded and split into chunks
3. Chunks are converted to embeddings via OpenAI API
4. Embeddings are stored in FAISS vector database
5. User asks a question
6. Question is converted to embedding
7. Most similar chunks are retrieved from FAISS
8. Context + question sent to OpenAI GPT
9. Response returned with source citations

## Security Notes

- API keys are handled as environment variables
- No data persistence beyond session
- Files are processed in temporary directories
- No external network access except OpenAI API

## License

MIT License - feel free to use and modify as needed.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## Support

For issues and questions:
- Check the troubleshooting section
- Review logs in the Docker container
- Open an issue with detailed error messages
