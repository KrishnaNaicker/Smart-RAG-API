import os
import requests
from io import BytesIO
import logging
import uuid
import re
from typing import List, Dict, Any
import numpy as np
from collections import defaultdict

# --- FastAPI Imports ---
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

# --- Core Libraries ---
import fitz  # PyMuPDF for PDF processing
from sentence_transformers import SentenceTransformer
import faiss  # For local vector storage (more reliable than Pinecone for this use case)
from transformers import pipeline
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

# --- Download required NLTK data ---
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

# --- Environment Variables ---
from dotenv import load_dotenv
load_dotenv()

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Configuration ---
EXPECTED_TOKEN = "4b11cc8862526987b9e5548e17ea3c9a864e2a3b54bd598ab9a8dc37af93a623"
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

class DocumentChunk:
    """Represents a chunk of document with metadata."""
    def __init__(self, content: str, page_num: int, chunk_id: str, metadata: Dict[str, Any] = None):
        self.content = content
        self.page_num = page_num
        self.chunk_id = chunk_id
        self.metadata = metadata or {}
        self.embedding = None

class SmartRAGSystem:
    """A robust RAG system with multiple retrieval strategies."""
    
    def __init__(self):
        # Initialize better embedding model
        logger.info("Loading embedding model...")
        self.embedding_model = SentenceTransformer('all-mpnet-base-v2')
        
        # Initialize QA model for verification
        logger.info("Loading QA model...")
        self.qa_pipeline = pipeline(
            "question-answering",
            model="deepset/roberta-base-squad2",
            tokenizer="deepset/roberta-base-squad2"
        )
        
        # Initialize text processing
        self.stop_words = set(stopwords.words('english'))
        
        # Storage for current session
        self.chunks = []
        self.embeddings = None
        self.faiss_index = None
        
    def clean_text(self, text: str) -> str:
        """Advanced text cleaning for better processing."""
        if not text:
            return ""
            
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Fix common PDF extraction issues
        text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)  # Fix hyphenated words
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)  # Add space between camelCase
        
        # Normalize punctuation
        text = re.sub(r'["""]', '"', text)
        text = re.sub(r"[\'']", "'", text)
        text = re.sub(r'[–—]', '-', text)
        
        # Remove extra newlines but preserve paragraph structure
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r'\n\s*\n', '\n\n', text)
        
        return text.strip()
    
    def extract_pdf_content(self, pdf_url: str) -> List[Dict[str, Any]]:
        """Extract content from PDF with improved structure preservation."""
        logger.info(f"Downloading PDF from: {pdf_url}")
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(pdf_url, headers=headers, timeout=30)
        response.raise_for_status()
        
        pdf_stream = BytesIO(response.content)
        pdf_doc = fitz.open(stream=pdf_stream, filetype="pdf")
        
        pages_content = []
        
        for page_num in range(len(pdf_doc)):
            page = pdf_doc[page_num]
            
            # Extract text with multiple methods for robustness
            text_dict = page.get_text("dict")
            raw_text = page.get_text()
            
            # Process structured text
            cleaned_text = self.clean_text(raw_text)
            
            if cleaned_text.strip():
                pages_content.append({
                    'page_num': page_num + 1,
                    'content': cleaned_text,
                    'char_count': len(cleaned_text),
                    'word_count': len(cleaned_text.split())
                })
        
        pdf_doc.close()
        logger.info(f"Extracted content from {len(pages_content)} pages")
        return pages_content
    
    def create_smart_chunks(self, pages_content: List[Dict[str, Any]]) -> List[DocumentChunk]:
        """Create intelligent chunks that preserve context."""
        chunks = []
        chunk_id_counter = 0
        
        for page_data in pages_content:
            page_num = page_data['page_num']
            content = page_data['content']
            
            # Split into sentences for better chunking
            sentences = sent_tokenize(content)
            
            # Group sentences into chunks of optimal size (increased for better context)
            current_chunk = []
            current_length = 0
            target_chunk_size = 600  # Increased from 400 for better context
            
            for sentence in sentences:
                sentence_length = len(sentence)
                
                # If adding this sentence would make chunk too large, save current chunk
                if current_length + sentence_length > target_chunk_size and current_chunk:
                    chunk_content = ' '.join(current_chunk)
                    if len(chunk_content.strip()) > 50:  # Only meaningful chunks
                        chunks.append(DocumentChunk(
                            content=chunk_content,
                            page_num=page_num,
                            chunk_id=f"chunk_{chunk_id_counter}",
                            metadata={
                                'sentence_count': len(current_chunk),
                                'char_count': current_length
                            }
                        ))
                        chunk_id_counter += 1
                    
                    current_chunk = [sentence]
                    current_length = sentence_length
                else:
                    current_chunk.append(sentence)
                    current_length += sentence_length
            
            # Don't forget the last chunk
            if current_chunk:
                chunk_content = ' '.join(current_chunk)
                if len(chunk_content.strip()) > 50:
                    chunks.append(DocumentChunk(
                        content=chunk_content,
                        page_num=page_num,
                        chunk_id=f"chunk_{chunk_id_counter}",
                        metadata={
                            'sentence_count': len(current_chunk),
                            'char_count': current_length
                        }
                    ))
                    chunk_id_counter += 1
        
        logger.info(f"Created {len(chunks)} intelligent chunks")
        return chunks
    
    def build_vector_index(self, chunks: List[DocumentChunk]):
        """Build FAISS vector index for fast similarity search."""
        logger.info("Building vector embeddings...")
        
        # Extract text content for embedding
        texts = [chunk.content for chunk in chunks]
        
        # Create embeddings in batches for efficiency
        embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
        
        # Store embeddings in chunks
        for i, chunk in enumerate(chunks):
            chunk.embedding = embeddings[i]
        
        # Build FAISS index
        dimension = embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        self.faiss_index.add(embeddings.astype('float32'))
        
        self.chunks = chunks
        self.embeddings = embeddings
        
        logger.info(f"Built vector index with {len(chunks)} chunks")
    
    def enhanced_retrieval(self, query: str, top_k: int = 12) -> List[DocumentChunk]:
        """Enhanced retrieval with better keyword and semantic matching."""
        if not self.chunks or self.faiss_index is None:
            return []
        
        # 1. Semantic retrieval using embeddings
        query_embedding = self.embedding_model.encode([query])
        faiss.normalize_L2(query_embedding)
        
        scores, indices = self.faiss_index.search(query_embedding.astype('float32'), min(top_k * 2, len(self.chunks)))
        
        semantic_chunks = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.chunks):
                chunk = self.chunks[idx]
                chunk.semantic_score = float(score)
                semantic_chunks.append(chunk)
        
        # 2. Enhanced keyword-based retrieval with phrase matching
        query_lower = query.lower()
        query_words = set(word_tokenize(query_lower)) - self.stop_words
        
        # Extract important phrases and numbers
        number_pattern = r'\b\d+\s*(?:days?|months?|years?)\b'
        query_numbers = re.findall(number_pattern, query_lower)
        
        keyword_scores = defaultdict(float)
        
        for i, chunk in enumerate(self.chunks):
            chunk_lower = chunk.content.lower()
            chunk_words = set(word_tokenize(chunk_lower)) - self.stop_words
            
            # Calculate keyword overlap score
            common_words = query_words.intersection(chunk_words)
            base_score = len(common_words) / max(len(query_words), 1) if query_words else 0
            
            # Boost score for exact phrase matches
            phrase_boost = 0
            if len(query.split()) > 1:
                if query_lower in chunk_lower:
                    phrase_boost = 0.5
            
            # Boost score for number/time period matches
            number_boost = 0
            for num_phrase in query_numbers:
                if num_phrase in chunk_lower:
                    number_boost = 0.3
            
            # Special keyword boosts for insurance-specific terms
            special_keywords = {
                'grace period': ['grace', 'premium', 'payment', 'due'],
                'waiting period': ['waiting', 'pre-existing', 'disease', 'commencement'],
                'exclusion': ['exclude', 'exclusion', 'not covered', 'except'],
                'coverage': ['cover', 'benefit', 'include', 'treatment'],
                'premium': ['premium', 'payment', 'due', 'amount']
            }
            
            special_boost = 0
            for key_phrase, related_words in special_keywords.items():
                if key_phrase in query_lower:
                    matching_related = sum(1 for word in related_words if word in chunk_lower)
                    if matching_related > 0:
                        special_boost = min(0.2 * matching_related, 0.4)
            
            if base_score > 0 or phrase_boost > 0 or number_boost > 0 or special_boost > 0:
                keyword_scores[i] = base_score + phrase_boost + number_boost + special_boost
        
        # 3. Combine scores and rank
        final_chunks = {}
        
        # Add semantic chunks
        for chunk in semantic_chunks:
            chunk_idx = self.chunks.index(chunk)
            final_chunks[chunk_idx] = {
                'chunk': chunk,
                'semantic_score': chunk.semantic_score,
                'keyword_score': keyword_scores.get(chunk_idx, 0.0)
            }
        
        # Add high-scoring keyword chunks
        for chunk_idx, keyword_score in keyword_scores.items():
            if keyword_score > 0.15 and chunk_idx not in final_chunks:  # Lowered threshold
                final_chunks[chunk_idx] = {
                    'chunk': self.chunks[chunk_idx],
                    'semantic_score': 0.0,
                    'keyword_score': keyword_score
                }
        
        # Calculate combined score and sort
        for chunk_data in final_chunks.values():
            # Give more weight to keyword matching for specific queries
            semantic_weight = 0.6 if len(query_words) < 3 else 0.5
            keyword_weight = 1.0 - semantic_weight
            
            combined_score = (semantic_weight * chunk_data['semantic_score'] + 
                            keyword_weight * chunk_data['keyword_score'])
            chunk_data['combined_score'] = combined_score
        
        # Sort by combined score and return top chunks
        sorted_chunks = sorted(final_chunks.values(), 
                             key=lambda x: x['combined_score'], 
                             reverse=True)
        
        return [item['chunk'] for item in sorted_chunks[:top_k]]
    
    def answer_question_with_context(self, question: str, context_chunks: List[DocumentChunk]) -> str:
        """Answer question using retrieved context with improved approach."""
        if not context_chunks:
            return "No relevant information found in the document to answer this question."
        
        # Combine context from multiple chunks
        context_parts = []
        for i, chunk in enumerate(context_chunks):
            context_parts.append(f"[Context {i+1} - Page {chunk.page_num}] {chunk.content}")
        
        combined_context = "\n\n".join(context_parts)
        
        # Limit context size but be more generous
        if len(combined_context) > 4000:
            combined_context = combined_context[:4000] + "..."
        
        # First try the transformer QA pipeline for short, direct answers
        try:
            result = self.qa_pipeline(question=question, context=combined_context)
            
            # Check if the answer is too short or just a fragment
            answer = result['answer'].strip()
            confidence = result['score']
            
            # If confidence is reasonable and answer looks complete, use it
            if confidence > 0.3 and len(answer) > 5:
                # Check if it's a complete sentence or meaningful phrase
                if any(char in answer for char in '.!?') or len(answer.split()) > 2:
                    return answer
            
        except Exception as e:
            logger.error(f"Error in QA pipeline: {e}")
        
        # Always use LLM for more complete answers
        return self.generate_answer_with_llm(question, combined_context)
    
    def generate_answer_with_llm(self, question: str, context: str) -> str:
        """Generate answer using Google Generative AI with improved prompting."""
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            from langchain.prompts import ChatPromptTemplate
            
            llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash-latest", 
                temperature=0.0,
                google_api_key=GOOGLE_API_KEY
            )
            
            prompt_template = """
You are an expert at extracting precise information from insurance policy documents. Answer the question based ONLY on the provided context.

Context from document:
{context}

Question: {question}

Instructions:
1. Provide a complete, specific answer that directly addresses the question
2. Include relevant details like time periods, amounts, or conditions mentioned in the context
3. If the question asks about a time period (grace period, waiting period), include the exact duration
4. If the question asks about exclusions or coverage, provide a clear, concise summary
5. Write your answer as a complete statement, not just isolated facts
6. If multiple relevant details are found, combine them into a coherent response
7. If the exact information is not in the context, state clearly that it's not available

Answer:"""
            
            prompt = ChatPromptTemplate.from_template(prompt_template)
            chain = prompt | llm
            
            response = chain.invoke({
                "context": context,
                "question": question
            })
            
            answer = response.content.strip()
            
            # Post-process the answer to ensure it's well-formed
            if answer and not answer.endswith('.'):
                if not any(answer.endswith(punct) for punct in ['!', '?', ':']):
                    answer += '.'
            
            return answer
            
        except Exception as e:
            logger.error(f"Error with LLM generation: {e}")
            return "Unable to generate answer due to processing error."
    
    def process_document_and_answer(self, pdf_url: str, questions: List[str]) -> List[str]:
        """Main method to process document and answer questions."""
        try:
            # 1. Extract PDF content
            pages_content = self.extract_pdf_content(pdf_url)
            
            # 2. Create intelligent chunks
            chunks = self.create_smart_chunks(pages_content)
            
            if not chunks:
                return ["No content could be extracted from the document."] * len(questions)
            
            # 3. Build vector index
            self.build_vector_index(chunks)
            
            # 4. Answer each question
            answers = []
            for question in questions:
                logger.info(f"Processing question: {question[:100]}...")
                
                # Retrieve relevant chunks with enhanced method
                relevant_chunks = self.enhanced_retrieval(question, top_k=10)
                
                # Generate answer
                answer = self.answer_question_with_context(question, relevant_chunks)
                answers.append(answer)
                
                logger.info(f"Answer generated: {answer[:100]}...")
            
            return answers
            
        except Exception as e:
            logger.error(f"Error processing document: {e}")
            return [f"Error processing question: {str(e)}"] * len(questions)

# --- Initialize the RAG system ---
rag_system = SmartRAGSystem()

# --- Pydantic Models ---
class QueryRequest(BaseModel):
    documents: str = Field(..., description="URL of the document to process")
    questions: List[str]

class AnswerResponse(BaseModel):
    answers: List[str]

# --- FastAPI App ---
app = FastAPI(title="Smart RAG API", description="Robust Document Q&A System")

@app.post("/hackrx/run", response_model=AnswerResponse)
async def run_submission(request: QueryRequest, authorization: str = Header(None)):
    """Main endpoint for document processing and question answering."""
    
    # Verify authorization
    if not authorization or authorization.split(" ")[1] != EXPECTED_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")
    
    try:
        logger.info(f"Processing {len(request.questions)} questions for document: {request.documents}")
        
        # Process document and get answers
        answers = rag_system.process_document_and_answer(
            request.documents, 
            request.questions
        )
        
        logger.info("Successfully generated all answers")
        return AnswerResponse(answers=answers)
        
    except Exception as e:
        logger.error(f"Request processing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

@app.get("/")
def read_root():
    return {
        "status": "Smart RAG API is running",
        "description": "POST to /hackrx/run with document URL and questions"
    }

@app.get("/health")
def health_check():
    return {"status": "healthy", "embedding_model": "all-mpnet-base-v2"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)