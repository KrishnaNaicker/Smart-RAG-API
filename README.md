# Smart RAG API

A robust Document Question-Answering system built with FastAPI, using advanced retrieval techniques and multiple AI models for accurate information extraction from PDF documents.

## Features

- **Smart PDF Processing**: Advanced text extraction and cleaning from PDF documents
- **Intelligent Chunking**: Context-aware document segmentation for better retrieval
- **Multi-Modal Retrieval**: Combines semantic search with keyword matching
- **Enhanced QA Pipeline**: Uses both transformer models and LLM for comprehensive answers
- **FastAPI Integration**: RESTful API with proper authentication and error handling

## Technologies Used

- **FastAPI**: Web framework for building the API
- **PyMuPDF (fitz)**: PDF text extraction
- **SentenceTransformers**: Semantic embeddings (all-mpnet-base-v2)
- **FAISS**: Vector similarity search
- **Transformers**: Question-answering pipeline (RoBERTa)
- **LangChain + Google Generative AI**: Advanced answer generation
- **NLTK**: Natural language processing utilities

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd hackrx
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
# Create a .env file with:
GOOGLE_API_KEY=your_google_api_key_here
```

4. Download NLTK data (automatically handled on first run):
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

## Usage

### Running the API

```bash
python app.py
```

The API will be available at `http://localhost:8000`

### API Endpoints

#### POST `/hackrx/run`

Process a document and answer questions about it.

**Headers:**
```
Authorization: Bearer 4b11cc8862526987b9e5548e17ea3c9a864e2a3b54bd598ab9a8dc37af93a623
```

**Request Body:**
```json
{
  "documents": "https://example.com/document.pdf",
  "questions": [
    "What is the grace period for premium payments?",
    "What are the exclusions mentioned in the policy?"
  ]
}
```

**Response:**
```json
{
  "answers": [
    "The grace period for premium payments is 30 days from the due date.",
    "The policy excludes pre-existing diseases, self-inflicted injuries, and war-related incidents."
  ]
}
```

#### GET `/health`

Check API health status.

#### GET `/`

Get basic API information.

## Architecture

### Core Components

1. **SmartRAGSystem**: Main class handling document processing and question answering
2. **DocumentChunk**: Data structure for document segments with metadata
3. **Enhanced Retrieval**: Multi-strategy approach combining semantic and keyword search
4. **Smart Chunking**: Context-aware document segmentation
5. **LLM Integration**: Google Generative AI for comprehensive answer generation

### Processing Pipeline

1. **PDF Extraction**: Download and extract text from PDF documents
2. **Text Cleaning**: Advanced preprocessing to handle PDF artifacts
3. **Smart Chunking**: Sentence-aware chunking with optimal size management
4. **Vector Indexing**: Build FAISS index for semantic similarity search
5. **Enhanced Retrieval**: Multi-modal retrieval combining multiple strategies
6. **Answer Generation**: Use both transformer QA and LLM for comprehensive answers

## Configuration

### Environment Variables

- `GOOGLE_API_KEY`: Required for LLM-based answer generation
- Token authentication is hardcoded for security (production apps should use proper auth)

### Model Configuration

- **Embedding Model**: `all-mpnet-base-v2` (can be changed in SmartRAGSystem.__init__)
- **QA Model**: `deepset/roberta-base-squad2`
- **LLM Model**: `gemini-1.5-flash-latest`

## Development

### Project Structure

```
hackrx/
├── app.py              # Main application file
├── requirements.txt    # Python dependencies
├── .env               # Environment variables (create this)
├── .gitignore         # Git ignore rules
└── README.md          # This file
```

### Key Features

- **Robust Error Handling**: Comprehensive error management throughout the pipeline
- **Logging**: Detailed logging for debugging and monitoring
- **Scalable Architecture**: Modular design for easy extension
- **Production Ready**: FastAPI with proper HTTP status codes and validation

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built for HackRX competition
- Uses state-of-the-art NLP models for accurate information retrieval
- Optimized for insurance document processing but adaptable to other domains
