from typing import TypedDict, List, Dict, Any, Optional
import asyncio
from concurrent.futures import ThreadPoolExecutor
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage, SystemMessage
from PyPDF2 import PdfReader
import chromadb
from chromadb.utils import embedding_functions
import uuid
import json
import operator
import logging
import time
from functools import wraps
import re
import hashlib
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up comprehensive logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# State Management with proper annotations
class PDFQAState(TypedDict):
    # Input states
    pdf_file: Optional[Any]
    profession: str
    question_count: int
    custom_prompt: str
    user_prompt: Optional[str]
    mcqs: Optional[List[Dict]]
    one_lines: Optional[List[Dict]]
    session_id: Optional[str]  # Added for progress tracking
    # Processing states
    extracted_text: str
    chunks: List[str]
    collection: Optional[Any]
    # Output states
    generated_mcqs: str
    generated_one_lines: str
    evaluation_results: Dict[str, Any]
    error_message: str
    # Progress tracking
    current_progress: int
    progress_message: str

# Initialize components with rate limit handling
# Initialize components with rate limit handling - API key from environment
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in environment variables. Please set it in .env file")

groq_llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model="llama-3.1-8b-instant",
    temperature=0.7,
    max_tokens=1024,
    timeout=60,
    max_retries=1
)

chroma_client = chromadb.Client()
embedding_fn = embedding_functions.DefaultEmbeddingFunction()

# Global collection variable
current_collection = None

# Evaluation cache to avoid repeat API calls
evaluation_cache = {}

# Enhanced progress tracking
progress_data = {}

def update_progress(session_id, progress, status, **details):
    """Update progress with detailed information"""
    if not session_id:
        return

    # Get existing progress data to preserve previous values
    existing_data = progress_data.get(session_id, {}).get("details", {})

    # Merge new details with existing ones (preserve previous data)
    merged_details = {
        "total_pages": details.get("total_pages", existing_data.get("total_pages", 0)),
        "chunk_count": details.get("chunk_count", existing_data.get("chunk_count", 0)),
        "total_questions": details.get("total_questions", existing_data.get("total_questions", 0)),
        "mcq_generated": details.get("mcq_generated", existing_data.get("mcq_generated", 0)),
        "short_generated": details.get("short_generated", existing_data.get("short_generated", 0))}

    progress_data[session_id] = {
        "progress": progress,
        "status": status,
        "details": merged_details,
        "timestamp": time.time()
    }
    logger.info(f"Progress {session_id}: {progress}% - {status}")

# Rate limit handling decorator (keep existing)
def retry_with_exponential_backoff(
    initial_delay: float = 2,
    exponential_base: float = 2,
    jitter: bool = True,
    max_retries: int = 4):
    """Retry a function with exponential backoff for rate limits."""
    
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            num_retries = 0
            delay = initial_delay

            while True:
                try:
                    return await func(*args, **kwargs)
                    
                except Exception as e:
                    error_str = str(e)
                    
                    # Check if it's a rate limit error
                    if any(keyword in error_str.lower() for keyword in [
                        "rate_limit", "rate limit", "429", "too many requests", 
                        "tokens per minute", "tpm"
                    ]):
                        num_retries += 1
                        
                        if num_retries > max_retries:
                            logger.error(f"Maximum retries ({max_retries}) exceeded for rate limit")
                            return {
                                "status": "partial", 
                                "explanation": "Rate limit exceeded - please try again later",
                                "correct_options": ["A"]
                            }
                        
                        # Extract wait time from error message if available
                        wait_time = delay
                        if "try again in" in error_str:
                            try:
                                time_part = error_str.split("try again in")[1].split(".")[0]
                                if "m" in time_part and "s" in time_part:
                                    minutes = int(time_part.split("m")[0])
                                    seconds = int(time_part.split("m")[1].replace("s", ""))
                                    wait_time = minutes * 60 + seconds + 2
                                elif "s" in time_part:
                                    wait_time = float(time_part.replace("s", "")) + 2
                            except:
                                wait_time = delay
                        
                        logger.info(f"Rate limit hit. Waiting {wait_time:.1f}s before retry {num_retries}/{max_retries}")
                        await asyncio.sleep(wait_time)
                        
                        # Exponential backoff for next retry
                        delay *= exponential_base
                        if jitter:
                            delay += (delay * 0.1 * (num_retries % 3))
                            
                    else:
                        # Not a rate limit error, re-raise
                        logger.error(f"Non-rate-limit error in {func.__name__}: {e}")
                        raise e
                        
        return wrapper
    return decorator

# Cache management (keep existing functions)
def get_cache_key(question: str, answer: str, context: str) -> str:
    """Generate cache key for evaluation results"""
    content = f"{question[:100]}_{answer[:100]}_{context[:100]}"
    return hashlib.md5(content.encode()).hexdigest()

def get_cached_result(cache_key: str):
    """Get cached evaluation result"""
    return evaluation_cache.get(cache_key)

def cache_result(cache_key: str, result: Dict):
    """Cache evaluation result"""
    evaluation_cache[cache_key] = result
    
    # Keep cache size manageable
    if len(evaluation_cache) > 1000:
        # Remove oldest entries
        keys_to_remove = list(evaluation_cache.keys())[:200]
        for key in keys_to_remove:
            del evaluation_cache[key]

# Enhanced Node Functions with Progress Tracking
def extract_pdf_text(state: PDFQAState) -> PDFQAState:
    """Extract text from uploaded PDF file with smart content detection"""
    try:
        session_id = state.get("session_id")
        pdf_file = state.get("pdf_file")
        custom_prompt = state.get("custom_prompt", "")
        
        if not pdf_file:
            if session_id:
                update_progress(session_id, 0, "Error: No PDF file provided")
            return {"error_message": "No PDF file provided", "current_progress": 0}

        # Ensure file pointer is at the beginning
        if hasattr(pdf_file, 'seek'):
            pdf_file.seek(0)
        elif hasattr(pdf_file, 'stream'):
            pdf_file.stream.seek(0)

        reader = PdfReader(pdf_file)
        total_pages = len(reader.pages)
        
        # Smart content detection
        start_page = detect_smart_content_start(reader, custom_prompt)
        actual_pages_to_process = total_pages - start_page
        
        # Update progress - starting extraction
        if start_page > 0:
            update_progress(session_id, 5, f"Auto-detected content starts at page {start_page + 1}, extracting {actual_pages_to_process} pages...")
        else:
            update_progress(session_id, 5, f"Extracting all {total_pages} pages (content detected from page 1)...")

        text = ""
        
        # Extract text from detected content pages
        for i in range(start_page, total_pages):
            page_text = reader.pages[i].extract_text() or ""
            text += page_text + "\n"
            
            # Update progress every 10 pages or on completion
            if session_id and (i % 10 == 0 or i == total_pages - 1):
                pages_processed = i - start_page + 1
                page_progress = 10 + (pages_processed * 15 // actual_pages_to_process)
                update_progress(
                    session_id,
                    page_progress,
                    f"Extracted page {i + 1} ({pages_processed}/{actual_pages_to_process} content pages)",
                    pages_processed=pages_processed,
                    total_pages=actual_pages_to_process,
                    text_length=len(text)
                )

        if not text.strip():
            if session_id:
                update_progress(session_id, 0, "Error: No extractable text found in content pages")
            return {"error_message": "No extractable text found in content pages", "current_progress": 0}

        # Final extraction progress update
        if session_id:
            update_progress(
                session_id,
                25,
                f"Smart extraction complete: {len(text):} characters from {actual_pages_to_process} pages",
                total_pages=actual_pages_to_process,
                text_length=len(text),
                extraction_complete=True,
                pages_excluded=start_page,
                content_start_page=start_page + 1
            )

        logger.info(f"Smart extraction: pages {start_page + 1}-{total_pages} ({actual_pages_to_process} content pages)")

        return {
            "extracted_text": text,
            "error_message": "",
            "current_progress": 25,
            "progress_message": "Smart content extraction completed",
            "session_id": session_id
        }

    except Exception as e:
        if session_id:
            update_progress(session_id, 0, f"PDF extraction failed: {str(e)}")
        logger.error(f"PDF extraction error: {str(e)}")
        return {"error_message": f"PDF extraction error: {str(e)}", "current_progress": 0}

def clean_content_text(text: str) -> str:
    """Remove document structure references from the source text itself"""
    # Remove common structural elements before processing
    structural_patterns = [
        r'Chapter\s+\d+[:\-\s]*[^\n]*\n',  # Chapter headers
        r'Section\s+\d+\.?\d*[:\-\s]*[^\n]*\n',  # Section headers  
        r'Part\s+[IVX\d]+[:\-\s]*[^\n]*\n',  # Part headers
        r'Page\s+\d+\s*\n',  # Page numbers
        r'Table\s+of\s+Contents[^\n]*\n',  # TOC references
        r'See\s+(page|chapter|section)\s+\d+',  # Cross-references
        r'Refer\s+to\s+(page|chapter|section)\s+\d+',  # References
        r'^\d+\.\d+\.\d+\s+',  # Numbering like 1.2.3
        r'^\d+\.\d+\s+',  # Numbering like 1.2
    ]
    
    cleaned_text = text
    for pattern in structural_patterns:
        cleaned_text = re.sub(pattern, '', cleaned_text, flags=re.MULTILINE | re.IGNORECASE)
    
    # Remove excessive whitespace
    cleaned_text = re.sub(r'\n{3}', '\n\n', cleaned_text)
    cleaned_text = re.sub(r'[ \t]{2}', ' ', cleaned_text)
    
    return cleaned_text.strip()

def detect_smart_content_start(reader: PdfReader, custom_prompt: str = "") -> int:
    """Intelligently detect where actual content starts in the PDF"""
    total_pages = len(reader.pages)
    max_scan_pages = min(15, total_pages)  # Don't scan more than 15 pages
    
    # 1. Check for explicit user instruction in custom prompt
    if custom_prompt:
        explicit_start = parse_explicit_page_instruction(custom_prompt)
        if explicit_start is not None:
            logger.info(f"Using explicit user instruction: start from page {explicit_start + 1}")
            return explicit_start
    
    # 2. Analyze each page to find content
    page_scores = []
    
    for i in range(max_scan_pages):
        page_text = reader.pages[i].extract_text() or ""
        page_text_lower = page_text.lower()
        
        # Calculate content indicators
        score = calculate_content_score(page_text, page_text_lower)
        page_scores.append((i, score, len(page_text)))
        
        logger.debug(f"Page {i + 1}: score={score:.1f}, length={len(page_text)}")
    
    # 3. Find the first page that looks like real content
    content_threshold = 7.0  # Minimum score to be considered content
    
    for page_idx, score, text_length in page_scores:
        if score >= content_threshold:
            logger.info(f"Content detected starting at page {page_idx + 1} (score: {score:.1f})")
            return page_idx
    
    # 4. Fallback: Use the page with the highest score if above minimum threshold
    if page_scores:
        best_page = max(page_scores, key=lambda x: x[1])
        page_idx, score, text_length = best_page
        
        if score >= 4.0 and text_length > 100:  # Lower threshold for fallback
            logger.info(f"Using best content page: {page_idx + 1} (score: {score:.1f})")
            return page_idx
    
    # 5. Ultimate fallback: start from page 1
    logger.info("No clear content pattern detected, starting from page 1")
    return 0

def calculate_content_score(page_text: str, page_text_lower: str) -> float:
    """Calculate a score indicating how likely a page contains actual content"""
    
    # Start with base score
    score = 0.0
    
    # Text length indicators
    word_count = len(page_text.split())
    char_count = len(page_text.strip())
    
    if word_count >= 200:
        score += 3.0
    elif word_count >= 100:
        score += 2.0
    elif word_count >= 50:
        score += 1.0
    
    # Paragraph structure (good content has multiple paragraphs)
    paragraph_count = len([p for p in page_text.split('\n\n') if p.strip()])
    if paragraph_count >= 3:
        score += 2.0
    elif paragraph_count >= 2:
        score += 1.0
    
    # Sentence structure (good content has proper sentences)
    sentence_count = page_text.count('.') + page_text.count('!') + page_text.count('?')
    if sentence_count >= 10:
        score += 2.0
    elif sentence_count >= 5:
        score += 1.0
    
    # Check for educational/content keywords (GOOD indicators)
    content_keywords = [
        'the', 'and', 'of', 'in', 'to', 'for', 'with', 'by', 'from', 'about',
        'this', 'that', 'these', 'those', 'can', 'will', 'should', 'must',
        'chapter', 'section', 'example', 'figure', 'table', 'definition',
        'introduction', 'conclusion', 'method', 'result', 'analysis'
    ]
    
    content_keyword_count = sum(1 for keyword in content_keywords if keyword in page_text_lower)
    if content_keyword_count >= 15:
        score += 2.0
    elif content_keyword_count >= 10:
        score += 1.0
    
    # Check for front matter keywords (BAD indicators - subtract score)
    front_matter_keywords = [
        'table of contents', 'contents', 'acknowledgment', 'acknowledgement',
        'preface', 'foreword', 'copyright', 'published by', 'isbn',
        'all rights reserved', 'publisher', 'edition', 'printing',
        'library of congress', 'cataloging', 'bibliography only',
        'index only', 'about the author', 'about this book',
        'college name', 'university name', 'department of',
        'faculty', 'staff', 'administration'
    ]
    
    front_matter_matches = sum(1 for keyword in front_matter_keywords if keyword in page_text_lower)
    if front_matter_matches >= 3:
        score -= 4.0  # Strong penalty
    elif front_matter_matches >= 2:
        score -= 2.0
    elif front_matter_matches >= 1:
        score -= 1.0
    
    # Check for table of contents pattern (lines with page numbers)
    toc_pattern_count = len(re.findall(r'\.{3}\s*\d+', page_text))  # Pattern like "Introduction...........5"
    if toc_pattern_count >= 3:
        score -= 3.0  # This is likely a table of contents
    
    # Very short pages are likely not main content
    if word_count < 20:
        score -= 2.0
    
    return score

def parse_explicit_page_instruction(custom_prompt: str) -> int:
    """Parse explicit page start instructions from user prompt"""
    prompt_lower = custom_prompt.lower()
    
    # Pattern: "start from page N"
    start_match = re.search(r'start\s+from\s+page\s+(\d+)', prompt_lower)
    if start_match:
        return int(start_match.group(1)) - 1  # Convert to 0-based index
    
    # Pattern: "skip first N pages"
    skip_match = re.search(r'skip\s+(?:first|initial)\s+(\d+)\s+pages?', prompt_lower)
    if skip_match:
        return int(skip_match.group(1))
    
    # Pattern: "exclude first N pages"
    exclude_match = re.search(r'exclude\s+(?:first|initial)\s+(\d+)\s+pages?', prompt_lower)
    if exclude_match:
        return int(exclude_match.group(1))
    
    # Pattern: "content starts page N"
    content_match = re.search(r'content\s+starts?\s+(?:from\s+)?page\s+(\d+)', prompt_lower)
    if content_match:
        return int(content_match.group(1)) - 1
        
    return None

def create_text_chunks(state: PDFQAState) -> PDFQAState:
    """Split extracted text into manageable chunks with detailed progress tracking"""
    try:
        session_id = state.get("session_id")
        text = state.get("extracted_text", "")
        
        if not text:
            if session_id:
                update_progress(session_id, 25, "Error: No text to chunk")
            return {"error_message": "No text to chunk", "current_progress": 25}

        # Update progress - starting chunking
        if session_id:
            update_progress(session_id, 30, "Cleaning content and analyzing text for chunking...")

        cleaned_text = clean_content_text(text)
        logger.info(f"Content cleaned: {len(text)} -> {len(cleaned_text)} characters")

        words = text.split()
        chunk_size = 400
        chunks = [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]
        
        # Calculate token count for each individual chunk
        
        total_tokens = 0
        
        for i, chunk in enumerate(chunks):
            # Calculate tokens for this specific chunk
            chunk_words = len(chunk.split())
            chunk_tokens = int(chunk_words * 0.75)  # Your existing estimation formula
            
            # Token tracking removed
            total_tokens += chunk_tokens
            
            # Update progress every 5 chunks or on the last chunk
            if session_id and (i % 5 == 0 or i == len(chunks) - 1):
                chunk_progress = 30 + ((i + 1) * 10 // len(chunks))
                update_progress(
                    session_id,
                    chunk_progress,
                    f"Processing chunk {i + 1}/{len(chunks)}...",
                    chunks_processed=i+1  # Send processed chunks so far
                )

        # Final update with all chunk details
        if session_id:
            update_progress(
                session_id,
                40,
                f"Created {len(chunks)} chunks from {len(words):} words",
                chunk_count=len(chunks),
                word_count=len(words),
                chunk_size=chunk_size,  # Complete list with all chunks
                chunking_complete=True
            )

        logger.info(f"Created {len(chunks)} chunks")
        
        return {
            "chunks": chunks,
            "current_progress": 40,
            "progress_message": f"Created {len(chunks)} content-focused chunks",
            "session_id": session_id,
            "token_estimate": total_tokens}

    except Exception as e:
        if session_id:
            update_progress(session_id, 30, f"Chunking failed: {str(e)}")
        logger.error(f"Chunking error: {str(e)}")
        return {"error_message": f"Chunking error: {str(e)}", "current_progress": 30}

def setup_vector_database_optimized(state: PDFQAState) -> PDFQAState:
    """Optimized vector database setup with batch operations and progress tracking"""
    global current_collection
    try:
        session_id = state.get("session_id")
        chunks = state.get("chunks", [])
        
        if not chunks:
            if session_id:
                update_progress(session_id, 40, "Error: No chunks to process")
            return {"error_message": "No chunks to process", "current_progress": 40}
        
        # Update progress - setting up database
        if session_id:
            update_progress(session_id, 45, "Setting up vector database...")
        
        # Create unique collection ID to avoid conflicts
        collection_id = f"pdf_chunks_{abs(hash(str(chunks[:3])))}"
        
        try:
            collection = chroma_client.get_collection(collection_id)
            current_collection = collection
            
            if session_id:
                update_progress(
                    session_id, 
                    55, 
                    "Using existing vector database",
                    chunk_count=len(chunks),
                    database_status="reused"
                )
            
            logger.info(f"Using existing collection: {collection_id}")
            return {
                "collection": collection,
                "current_progress": 55,
                "progress_message": "Vector database ready",
                "session_id": session_id
            }
        except:
            pass
        
        # Create new collection
        try:
            chroma_client.delete_collection(collection_id)
        except Exception:
            pass
        
        if session_id:
            update_progress(session_id, 50, "Creating new vector database...")
        
        collection = chroma_client.get_or_create_collection(
            name=collection_id,
            embedding_function=embedding_fn
        )
        
        # BATCH INSERT: Add all chunks at once
        chunk_ids = [str(uuid.uuid4()) for _ in chunks]
        collection.add(documents=chunks, ids=chunk_ids)
        
        current_collection = collection
        
        # Update progress with database creation complete
        if session_id:
            update_progress(
                session_id, 
                60, 
                f"Vector database created with {len(chunks)} chunks",
                chunk_count=len(chunks),
                database_status="created",
                embeddings_created=len(chunks)
            )
        
        logger.info(f"Created new collection with {len(chunks)} chunks")
        
        return {
            "collection": collection,
            "current_progress": 60,
            "progress_message": f"Vector database created with {len(chunks)} chunks",
            "session_id": session_id
        }
        
    except Exception as e:
        if session_id:
            update_progress(session_id, 45, f"Vector DB setup failed: {str(e)}")
        logger.error(f"Vector DB setup error: {str(e)}")
        return {"error_message": f"Vector DB setup error: {str(e)}", "current_progress": 45}

async def generate_questions_parallel(state: PDFQAState) -> PDFQAState:
    """Generate questions with rate limit handling, detailed progress tracking, and quality filtering"""
    try:
        session_id = state.get("session_id")
        chunks = state.get("chunks", [])
        profession = state.get("profession", "general")
        question_count = state.get("question_count", 10)
        
        if not chunks:
            if session_id:
                update_progress(session_id, 60, "Error: No chunks available for question generation")
            return {"error_message": "No chunks available for question generation"}
        
        # Initialize tracking variables
        mcq_list = []
        one_line_list = []
        mcq_generated = 0
        oneline_generated = 0
        
        # ADD THIS: Quality filter function
        def filter_irrelevant_questions(questions_text: str, question_type: str = "mcq") -> str:
            """Filter out irrelevant meta-questions about document structure"""
            if not questions_text or not questions_text.strip():
                return ""
            
            # Patterns that indicate irrelevant questions
            forbidden_patterns = [
            # Direct structure references
            r'.*chapter\s*\d*.*',
            r'.*page\s*\d*.*',
            r'.*section\s*\d*.*',
            r'.*part\s*\d*.*',
            
            # Location questions
            r'.*which\s+(part|section|chapter|page).*',
            r'.*in\s+which\s+(part|section|chapter|page).*',
            r'.*where\s+(in|is).*',
            r'.*what\s+(page|chapter|part|section).*',
            
            # Document references
            r'.*in\s+the\s+(book|text|document|material).*',
            r'.*part\s+of\s+the\s+(book|text|document).*',
            r'.*section\s+of\s+the\s+(book|text|document).*',
            r'.*chapter\s+of\s+the\s+(book|text|document).*',
            
            # Organization questions
            r'.*organize.*',
            r'.*structure.*',
            r'.*table\s+of\s+contents.*',
            r'.*contents.*',
            r'.*index.*',
            r'.*appendix.*',
            
            # Reference questions
            r'.*refer\s+to.*',
            r'.*reference.*',
            r'.*mention.*\s+(page|chapter|section).*',
            r'.*discuss.*\s+(page|chapter|section).*',
            r'.*found\s+in.*',
            r'.*located\s+in.*',
            
            # Meta questions about the document
            r'.*this\s+(book|text|document|material).*',
            r'.*the\s+(book|text|document|material)\s+contains.*',
            r'.*how\s+many\s+(chapters|pages|sections).*',
            r'.*number\s+of\s+(chapters|pages|sections).*',
            
            # Position indicators
            r'.*first\s+(chapter|section|part).*',
            r'.*last\s+(chapter|section|part).*',
            r'.*next\s+(chapter|section|part).*',
            r'.*previous\s+(chapter|section|part).*',
            
            # Generic document structure
            r'.*beginning\s+of.*',
            r'.*end\s+of.*',
            r'.*start\s+of.*',
            r'.*introduction\s+chapter.*',
            r'.*conclusion\s+chapter.*'
            ]
            
            lines = questions_text.split('\n')
            filtered_lines = []
            questions_removed = 0
    
            current_question = []
            
            for line in lines:
                line = line.strip()
                if not line:
                    if current_question:
                        filtered_lines.extend(current_question)
                        current_question = []
                    filtered_lines.append("")
                    continue
                
                # Check if this line starts a new question
                if re.match(r'^\d+\.', line) or line.startswith('Q:'):
                    # Process previous question if exists
                    if current_question:
                        question_text = ' '.join(current_question).lower()
                        is_forbidden = any(re.search(pattern, question_text) for pattern in forbidden_patterns)
                        
                        if not is_forbidden:
                            filtered_lines.extend(current_question)
                        else:
                            questions_removed += 1
                            logger.info(f"Filtered out question: {current_question[0][:50]}...")
                        
                        current_question = []
                    
                    current_question = [line]
                else:
                    # This is a continuation of the current question (options, answers, etc.)
                    current_question.append(line)
            
            # Process the last question
            if current_question:
                question_text = ' '.join(current_question).lower()
                is_forbidden = any(re.search(pattern, question_text) for pattern in forbidden_patterns)
                
                if not is_forbidden:
                    filtered_lines.extend(current_question)
                else:
                    questions_removed += 1
                    logger.info(f"Filtered out question: {current_question[0][:50]}...")
            
            filtered_text = '\n'.join(filtered_lines).strip()
            
            # Remove excessive empty lines
            filtered_text = re.sub(r'\n{3}', '\n\n', filtered_text)
            
            if questions_removed > 0:
                logger.info(f"Ultra-strict filter removed {questions_removed} irrelevant questions")
            
            return filtered_text
        
        # Update progress - starting question generation
        if session_id:
            update_progress(
                session_id, 
                65, 
                f"Generating {question_count} questions of each type...",
                target_mcq=question_count,
                target_short=question_count,
                chunks_available=len(chunks)
            )
        
        # Process chunks in smaller batches with delays
        batch_size = 1
        chunk_batches = [chunks[i:i+batch_size] for i in range(0, len(chunks), batch_size)]
        
        for batch_idx, batch in enumerate(chunk_batches):
            if mcq_generated >= question_count and oneline_generated >= question_count:
                break
                
            mcq_needed = max(0, question_count - mcq_generated)
            oneline_needed = max(0, question_count - oneline_generated)
            
            if mcq_needed == 0 and oneline_needed == 0:
                break
            
            # Update progress during generation
            batch_progress = 65 + (batch_idx * 20 // len(chunk_batches))  # Save 5% for filtering
            if session_id:
                update_progress(
                    session_id,
                    batch_progress,
                    f"Processing batch {batch_idx + 1}/{len(chunk_batches)}...",
                    mcq_generated=mcq_generated,
                    short_generated=oneline_generated,
                    target_mcq=question_count,
                    target_short=question_count
                )
            
            try:
                mcq_part, one_line_part = await process_chunk_batch_async(
                    batch, profession, mcq_needed, oneline_needed, state.get("custom_prompt", "") 
                )
                
                # MODIFIED: Apply quality filtering to MCQs
                if mcq_part and mcq_generated < question_count:
                    # Filter out irrelevant questions
                    filtered_mcq = filter_irrelevant_questions(mcq_part, "mcq")
                    
                    if filtered_mcq.strip():  # Only add if there's content after filtering
                        mcq_list.append(filtered_mcq)
                        mcq_count = len([
                            line for line in filtered_mcq.split('\n')
                            if line.strip() and not line.strip().startswith(('A)', 'B)', 'C)', 'D)'))
                            and any(char.isalpha() for char in line)
                        ])
                        mcq_generated += mcq_count
                
                # MODIFIED: Apply quality filtering to one-line questions
                if one_line_part and oneline_generated < question_count:
                    # Filter out irrelevant questions
                    filtered_oneline = filter_irrelevant_questions(one_line_part, "oneline")
                    
                    if filtered_oneline.strip():  # Only add if there's content after filtering
                        one_line_list.append(filtered_oneline)
                        oneline_count = len([line for line in filtered_oneline.split('\n') if 'Q:' in line])
                        oneline_generated += oneline_count
                
                # Add delay between batches to avoid rate limits
                if batch_idx < len(chunk_batches) - 1:
                    await asyncio.sleep(1.0)
                    
            except Exception as e:
                logger.error(f"Error processing batch {batch_idx}: {e}")
                continue
        
        # ADDED: Final filtering progress update
        if session_id:
            update_progress(
                session_id,
                85,
                "Applying quality filters to remove irrelevant questions...",
                mcq_generated=mcq_generated,
                short_generated=oneline_generated
            )
        
        # Final progress update for question generation
        if session_id:
            update_progress(
                session_id, 
                90, 
                f"Question generation complete (filtered for quality)",
                mcq_generated=mcq_generated,
                short_generated=oneline_generated,
                total_questions=mcq_generated + oneline_generated,
                generation_complete=True
            )
        
        # Log filtering results
        logger.info(f"Quality filtering applied: Generated {mcq_generated} relevant MCQs and {oneline_generated} relevant short questions")
        
        return {
            "generated_mcqs": "\n\n".join(mcq_list),
            "generated_one_lines": "\n\n".join(one_line_list),
            "current_progress": 90,
            "progress_message": f"Generated {mcq_generated} MCQs and {oneline_generated} short questions (quality filtered)",
            "session_id": session_id
        }
        
    except Exception as e:
        if session_id:
            update_progress(session_id, 60, f"Question generation failed: {str(e)}")
        logger.error(f"Question generation error: {str(e)}")
        return {"error_message": f"Question generation error: {str(e)}"}

# Keep all existing async helper functions (generate_from_combined_text, process_chunk_batch_async, etc.)
async def generate_from_combined_text(combined_text: str, profession: str, mcq_to_generate: int, oneline_to_generate: int, custom_prompt: str = ""):
    """Generate questions with enhanced content-focused prompts and strict filtering"""
    if len(combined_text.split()) > 800:
        combined_text = " ".join(combined_text.split()[:800])

    role_map = {
        "11th": "High school teacher",
        "engineering": "Engineering professor", 
        "medical": "Medical professor",
        "law": "Law professor",
        "professional": "Corporate trainer",
        "general": "Teacher"
    }

    # ENHANCED SYSTEM PROMPT - Much more restrictive
    system_prompt = f"""You are a {role_map.get(profession, 'teacher')} creating educational questions from study material.

ABSOLUTE RULES - NEVER VIOLATE:
1. Create questions ONLY about the actual subject matter, concepts, facts, processes, and knowledge
2. NEVER ask about document organization, structure, location, or navigation
3. NEVER mention: chapters, pages, parts, sections, books, documents, text, material
4. NEVER ask WHERE information is found - only ask WHAT the information IS
5. Focus on: definitions, explanations, processes, causes, effects, examples, applications
6. Ask questions that test understanding of the content itself

FORBIDDEN QUESTION TYPES (NEVER create these):
- "What chapter covers..."
- "Which part discusses..."  
- "Where in the text..."
- "What page mentions..."
- "In which section..."
- "This topic is found in..."
- "The book organizes..."
- Any question referring to document structure

REQUIRED QUESTION TYPES (Always create these):
- "What is the definition of..."
- "How does [process] work..."
- "What are the main causes of..."
- "Which factor affects..."
- "What happens when..."
- "Why does [phenomenon] occur..."
"""

    # ENHANCED BASE PROMPT - More explicit instructions
    base_prompt = f"""
Create {mcq_to_generate} multiple-choice question(s) and {oneline_to_generate} short answer question(s).

CONTENT FOCUS REQUIREMENTS:
✅ Ask about concepts, definitions, processes, facts within the material
✅ Test understanding of "what", "how", "why" regarding the subject matter
✅ Focus on knowledge someone would gain from studying this content
✅ Create questions about causes, effects, examples, applications, principles

🚫 NEVER ask about document structure, organization, or location
🚫 NEVER use words: chapter, page, part, section, book, text, document, material
🚫 NEVER ask WHERE something is mentioned or discussed
🚫 NEVER reference the source or its organization

FORMAT (strict adherence required):
MCQs:
1. [Content question about the actual subject]?
A) [Factual option about the content]
B) [Factual option about the content]  
C) [Factual option about the content]
D) [Factual option about the content]

Short Questions:
1. Q: [Content question about the subject]? A: [Answer based on the content].

EXAMPLE OF GOOD QUESTIONS:
- "What is photosynthesis?" (NOT "Which chapter covers photosynthesis?")  
- "How does mitosis occur?" (NOT "Where is mitosis discussed?")
- "What causes diabetes?" (NOT "What page mentions diabetes?")
"""

    # Add custom instructions if provided
    if custom_prompt and custom_prompt.strip():
        base_prompt += f"\nAdditional Instructions: {custom_prompt.strip()}\n"
        logger.info(f"Using custom instructions: {custom_prompt[:100]}...")

    base_prompt += f"\nStudy Material Content: {combined_text}"

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=base_prompt)
    ]

    try:
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(None, lambda: groq_llm.invoke(messages))
        content = response.content if response.content else ""
    except Exception as e:
        logger.error(f"LLM generation error: {e}")
        return "", ""

    # Parse response
    mcq_part = ""
    one_line_part = ""
    if content:
        if "MCQs:" in content and "Short Questions:" in content:
            parts = content.split("Short Questions:")
            mcq_part = parts[0].replace("MCQs:", "").strip()
            one_line_part = parts[1].strip()
        elif "MCQs:" in content:
            mcq_part = content.replace("MCQs:", "").strip()
        else:
            mcq_part = content

    return mcq_part, one_line_part

async def process_chunk_batch_async(batch: List[str], profession: str, mcq_needed: int, oneline_needed: int, custom_prompt: str = ""):
    """Process a batch of chunks with rate limit awareness"""
    if not batch:
        return "", ""

    combined_text = " ".join(batch)
    if len(combined_text.split()) > 600:
        combined_text = " ".join(combined_text.split()[:600])

    mcq_to_generate = min(2, mcq_needed)
    oneline_to_generate = min(2, oneline_needed)

    return await generate_from_combined_text(combined_text, profession, mcq_to_generate, oneline_to_generate, custom_prompt)

def generate_questions(state: PDFQAState) -> PDFQAState:
    """Wrapper function to run async question generation with proper event loop handling"""
    try:
        # Try to use existing event loop if available
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If loop is already running, create a task
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, generate_questions_parallel(state))
                return future.result()
        else:
            # If no loop is running, use run_until_complete
            return loop.run_until_complete(generate_questions_parallel(state))
    except RuntimeError:
        # Fallback: Use asyncio.run (works with nest_asyncio)
        try:
            return asyncio.run(generate_questions_parallel(state))
        except Exception as e:
            session_id = state.get("session_id")
            if session_id:
                update_progress(session_id, 60, f"Question generation failed: {str(e)}")
            return {"error_message": f"Question generation error: {str(e)}"}
    except Exception as e:
        session_id = state.get("session_id")
        if session_id:
            update_progress(session_id, 60, f"Question generation failed: {str(e)}")
        return {"error_message": f"Async execution error: {str(e)}"}

# Enhanced Service class
class PDFQAService:
    def __init__(self):
        self.workflow = create_pdf_qa_workflow()
        self.evaluation_workflow = create_evaluation_workflow()
        self.cache = {}

    def generate_qa_from_pdf_with_progress(self, initial_state):
        """Generate Q&A from PDF file with enhanced progress tracking"""
        try:
            result = self.workflow.invoke(initial_state)
            session_id = initial_state.get("session_id")
        
            if session_id:
                # Get existing progress data to preserve all previous stats
                existing_progress = progress_data.get(session_id, {}).get("details", {})
                
                # Count actual questions generated
                mcq_text = result.get("generated_mcqs", "")
                oneline_text = result.get("generated_one_lines", "")
                
                # More accurate question counting
                mcq_count = len([line for line in mcq_text.split('\n') 
                            if line.strip() and not line.strip().startswith(('A)', 'B)', 'C)', 'D)'))
                            and '?' in line]) if mcq_text else 0
                
                short_count = len([line for line in oneline_text.split('\n') 
                                if 'Q:' in line]) if oneline_text else 0
            
            # Final completion update preserving ALL accumulated data
            update_progress(
                session_id,
                100,
                "✅ PDF Processing Complete!",
                total_pages=existing_progress.get("total_pages", 0),    # PRESERVE from extraction
                chunk_count=existing_progress.get("chunk_count", 0),   # PRESERVE from chunking
                token_estimate=existing_progress.get("token_estimate", 0),  # PRESERVE from chunking
                mcq_generated=mcq_count,
                short_generated=short_count,
                total_questions=mcq_count + short_count,
                processing_complete=True
            )
            
            final_result = {
                "mcqs": result.get("generated_mcqs", ""),
                "one_lines": result.get("generated_one_lines", ""),
                "error": result.get("error_message", ""),
                "progress": result.get("current_progress", 100)
            }

            return final_result
            
        except Exception as e:
            logger.error(f"Error in PDF processing workflow: {e}")
            session_id = initial_state.get("session_id")
            if session_id:
                update_progress(session_id, 0, f"Processing failed: {str(e)}")
            
            return {
                "mcqs": "",
                "one_lines": "",
                "error": f"Processing error: {str(e)}",
                "progress": 0
            }

    # Keep existing evaluate_student_answers method
    def evaluate_student_answers(self, mcqs, one_lines):
        """Evaluate student answers using sequential processing"""
        initial_state = {
            "mcqs": mcqs,
            "one_lines": one_lines,
            "pdf_file": None,
            "profession": "general",
            "question_count": 0,
            "user_prompt": None,
            "extracted_text": "",
            "chunks": [],
            "collection": None,
            "generated_mcqs": "",
            "generated_one_lines": "",
            "evaluation_results": {},
            "error_message": "",
            "current_progress": 80,
            "progress_message": "Starting evaluation..."
        }

        try:
            result = self.evaluation_workflow.invoke(initial_state)
            return result.get("evaluation_results", {})
        except Exception as e:
            logger.error(f"Error in evaluation workflow: {e}")
            return {"error": f"Evaluation error: {str(e)}"}

# Keep all existing workflow creation functions
def create_pdf_qa_workflow():
    workflow = StateGraph(PDFQAState)
    
    workflow.add_node("extract_text", extract_pdf_text)
    workflow.add_node("create_chunks", create_text_chunks)
    workflow.add_node("setup_vectordb", setup_vector_database_optimized)
    workflow.add_node("generate_qa", generate_questions)
    
    workflow.add_edge(START, "extract_text")
    workflow.add_edge("extract_text", "create_chunks")
    workflow.add_edge("create_chunks", "setup_vectordb")
    workflow.add_edge("setup_vectordb", "generate_qa")
    workflow.add_edge("generate_qa", END)
    
    return workflow.compile()

def create_evaluation_workflow():
    """Create separate workflow for evaluation"""
    workflow = StateGraph(PDFQAState)
    workflow.add_node("evaluate", evaluate_answers)
    workflow.add_edge(START, "evaluate")
    workflow.add_edge("evaluate", END)
    return workflow.compile()

# Keep existing evaluation functions
@retry_with_exponential_backoff(initial_delay=2, max_retries=4)
async def evaluate_single_item_with_retry(item: Dict, collection: Any, item_type: str, progress_callback=None):
    """Evaluate a single item with automatic retry on rate limits"""
    try:
        if item_type == "mcq":
            qid = item.get("id")
            question = item.get("question", "")
            options = item.get("options", {})
            selected = item.get("selected", [])
            
            # Create cache key
            cache_key = get_cache_key(question, str(selected), str(options))
            cached = get_cached_result(cache_key)
            if cached:
                return qid, cached

            # Limit question and option lengths to save tokens
            question = question[:150]
            query_text = question + " " + " ".join(options.get(k, "")[:30] for k in selected)
            
            # Retrieve context with smaller result set
            try:
                qres = collection.query(query_texts=[query_text or question], n_results=1)
                docs = qres.get("documents", [[]])[0]
                context = " ".join(docs)[:200] if docs else ""
            except Exception:
                context = ""

            # Prepare concise evaluation prompt
            eval_prompt = f"""Grade MCQ. JSON only:
{{"status":"correct|incorrect","correct_options":["A"],"explanation":"Brief reason"}}

Q: {question}
Options: A){options.get("A","")[:40]} B){options.get("B","")[:40]} C){options.get("C","")[:40]} D){options.get("D","")[:40]}
Selected: {", ".join(selected)}
Context: {context}"""

            # Make async LLM call
            messages = [
                SystemMessage(content="You are an examiner. Return valid JSON only."),
                HumanMessage(content=eval_prompt)
            ]

            loop = asyncio.get_event_loop()
            resp = await loop.run_in_executor(None, lambda: groq_llm.invoke(messages))
            content = resp.content if resp.content else "{}"

            try:
                parsed = json.loads(content.strip().strip("`"))
                # Ensure required fields
                if "status" not in parsed:
                    parsed["status"] = "partial"
                if "correct_options" not in parsed:
                    parsed["correct_options"] = ["A"]
                if "explanation" not in parsed:
                    parsed["explanation"] = "Evaluation completed"
            except Exception as json_error:
                logger.warning(f"JSON parsing error: {json_error}")
                parsed = {
                    "status": "partial",
                    "explanation": content[:100] if content else "Could not parse evaluation",
                    "correct_options": ["A"]
                }

        elif item_type == "one_line":
            qid = item.get("id")
            question = item.get("question", "")
            answer = item.get("answer", "")
            
            # Create cache key
            cache_key = get_cache_key(question, answer, "")
            cached = get_cached_result(cache_key)
            if cached:
                return qid, cached

            question = question[:150]
            answer = answer[:100]
            query_text = question + " " + answer

            # Retrieve context
            try:
                qres = collection.query(query_texts=[query_text or question], n_results=1)
                docs = qres.get("documents", [[]])[0]
                context = " ".join(docs)[:200] if docs else ""
            except Exception:
                context = ""

            eval_prompt = f"""Grade answer. JSON only:
{{"status":"correct|incorrect|partial","model_answer":"Expected","explanation":"Brief reason"}}

Q: {question}
Answer: {answer}
Context: {context}"""

            # Make async LLM call
            messages = [
                SystemMessage(content="You are an examiner. Return valid JSON only."),
                HumanMessage(content=eval_prompt)
            ]

            loop = asyncio.get_event_loop()
            resp = await loop.run_in_executor(None, lambda: groq_llm.invoke(messages))
            content = resp.content if resp.content else "{}"

            try:
                parsed = json.loads(content.strip().strip("`"))
                if "status" not in parsed:
                    parsed["status"] = "partial"
                if "model_answer" not in parsed:
                    parsed["model_answer"] = "Sample answer"
                if "explanation" not in parsed:
                    parsed["explanation"] = "Evaluation completed"
            except Exception as json_error:
                logger.warning(f"JSON parsing error: {json_error}")
                parsed = {
                    "status": "partial",
                    "explanation": content[:100] if content else "Could not parse evaluation",
                    "model_answer": "Unable to determine"
                }

        # Cache the result
        cache_result(cache_key, parsed)
        
        if progress_callback:
            progress_callback()

        return qid, parsed

    except Exception as e:
        logger.error(f"Evaluation error for {item.get('id', 'unknown')}: {e}")
        return item.get("id", "unknown"), {
            "status": "partial",
            "explanation": f"Evaluation error: {str(e)[:50]}",
            "correct_options": ["A"] if item_type == "mcq" else None,
            "model_answer": "Error occurred" if item_type == "one_line" else None
        }

async def evaluate_answers_sequential(state: PDFQAState) -> PDFQAState:
    """Evaluate answers sequentially to handle rate limits"""
    global current_collection
    
    try:
        collection = current_collection or state.get("collection")
        mcqs = state.get("mcqs", [])
        one_lines = state.get("one_lines", [])
        
        if not collection:
            return {"error_message": "No collection available for evaluation"}

        results = {"mcqs": {}, "one_lines": {}}
        total_items = len(mcqs) + len(one_lines)
        processed_items = 0

        def update_progress():
            nonlocal processed_items
            processed_items += 1
            progress = 80 + (processed_items * 20 // total_items)
            logger.info(f"Evaluation progress: {processed_items}/{total_items} ({progress}%)")

        logger.info(f"Starting sequential evaluation of {total_items} items")

        # Process MCQs sequentially with delays
        for i, item in enumerate(mcqs):
            try:
                qid, result = await evaluate_single_item_with_retry(
                    item, collection, "mcq", update_progress
                )
                results["mcqs"][qid] = result
                
                # Add delay between requests (except for last item)
                if i < len(mcqs) - 1:
                    await asyncio.sleep(1.2)  # 1.2 second delay
                    
            except Exception as e:
                logger.error(f"Error evaluating MCQ {item.get('id', 'unknown')}: {e}")
                results["mcqs"][item.get("id", f"mcq_{i}")] = {
                    "status": "partial",
                    "correct_options": ["A"],
                    "explanation": "Evaluation failed due to system error"
                }
                update_progress()

        # Process one-lines sequentially with delays
        for i, item in enumerate(one_lines):
            try:
                qid, result = await evaluate_single_item_with_retry(
                    item, collection, "one_line", update_progress
                )
                results["one_lines"][qid] = result
                
                # Add delay between requests (except for last item)
                if i < len(one_lines) - 1:
                    await asyncio.sleep(1.2)  # 1.2 second delay
                    
            except Exception as e:
                logger.error(f"Error evaluating one-line {item.get('id', 'unknown')}: {e}")
                results["one_lines"][item.get("id", f"oneline_{i}")] = {
                    "status": "partial",
                    "model_answer": "Unable to evaluate",
                    "explanation": "Evaluation failed due to system error"
                }
                update_progress()

        logger.info("Sequential evaluation completed successfully")
        
        return {
            "evaluation_results": results,
            "current_progress": 100,
            "progress_message": f"Evaluation complete: {total_items} questions processed"
        }

    except Exception as e:
        logger.error(f"Sequential evaluation error: {str(e)}")
        return {"error_message": f"Sequential evaluation error: {str(e)}"}

def evaluate_answers(state: PDFQAState) -> PDFQAState:
    # ... [Keep existing implementation] ...
    try:
        # Try to use existing event loop if available
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If loop is already running, create a task in a new thread
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, evaluate_answers_sequential(state))
                return future.result()
        else:
            # If no loop is running, use run_until_complete
            return loop.run_until_complete(evaluate_answers_sequential(state))
    except RuntimeError:
        # Fallback: Use asyncio.run (works with nest_asyncio)
        try:
            return asyncio.run(evaluate_answers_sequential(state))
        except Exception as e:
            logger.error(f"Evaluation error: {str(e)}")
            return {"error_message": f"Evaluation error: {str(e)}"}
    except Exception as e:
        logger.error(f"Async evaluation error: {str(e)}")
        return {"error_message": f"Async evaluation error: {str(e)}"}

# Flask Integration with enhanced progress tracking
from flask import Flask, request, render_template, jsonify, Response
import json
import threading
import time

app = Flask(__name__)
pdf_qa_service = PDFQAService()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/generate_qa", methods=["POST"])
def generate_qa():
    # Enhanced debug logging
    logger.info("=" * 60)
    logger.info("🔍 FLASK REQUEST ANALYSIS")
    logger.info(f"Request method: {request.method}")
    logger.info(f"Content-Type: {request.content_type}")
    logger.info(f"Form keys: {list(request.form.keys())}")
    logger.info(f"Files keys: {list(request.files.keys())}")
    
    # Flexible file detection
    file = None
    for field_name in ['pdf_file', 'file', 'upload', 'document']:
        if field_name in request.files:
            file = request.files[field_name]
            logger.info(f"✅ Found file using field name: '{field_name}'")
            break
    
    if not file and request.files:
        file = list(request.files.values())[0]
        logger.info(f"✅ Using first available file")
    
    if not file or file.filename == '':
        logger.error("❌ No file found in request")
        return jsonify({"mcq": "", "one_line": "⚠️ No PDF uploaded."})

    profession = request.form.get("profession", "general")
    question_count = int(request.form.get("questionCount", 10))
    custom_prompt = request.form.get("customPrompt", "").strip() 

    # Generate unique session ID for progress tracking
    session_id = str(uuid.uuid4())
    
    # Initialize progress tracking
    update_progress(session_id, 0, "Starting PDF processing...")

    logger.info(f"📄 Processing: {file.filename}")
    logger.info(f"👤 Profession: {profession}")
    logger.info(f"🔢 Questions: {question_count}")
    logger.info(f"🆔 Session: {session_id}")

    try:
        # Enhanced initial state with session tracking
        initial_state = {
            "pdf_file": file,
            "profession": profession,
            "question_count": question_count,
            "custom_prompt": custom_prompt,
            "session_id": session_id,
            "extracted_text": "",
            "chunks": [],
            "collection": None,
            "generated_mcqs": "",
            "generated_one_lines": "",
            "evaluation_results": {},
            "error_message": "",
            "user_prompt": None,
            "mcqs": None,
            "one_lines": None,
            "current_progress": 0,
            "progress_message": "Starting PDF processing..."
        }

        result = pdf_qa_service.generate_qa_from_pdf_with_progress(initial_state)
        # Enhanced token calculation
        file.seek(0)
        pdf_content = file.read()
        file.seek(0)
        
        # Convert bytes to string safely
        try:
            input_text = pdf_content.decode('utf-8', errors='ignore') if isinstance(pdf_content, bytes) else str(pdf_content)
        except:
            input_text = str(pdf_content)
        
        output_text = result["mcqs"] + result["one_lines"]
        
        # More accurate token estimation (words * 0.75 factor for Groq)
        input_words = len(input_text.split()) + len(custom_prompt.split()) if custom_prompt else len(input_text.split())
        output_words = len(output_text.split())
        
        input_tokens = int(input_words * 0.75)
        output_tokens = int(output_words * 0.75)
        total_tokens = input_tokens + output_tokens
        estimated_cost = total_tokens * 0.00002  # Groq pricing

        if result["error"]:
            logger.error(f"Processing error: {result['error']}")
            return jsonify({
                "mcq": "", 
                "one_line": f"⚠️ {result['error']}", 
                "session_id": session_id
            })

        logger.info("✅ Successfully generated questions")
        return jsonify({
            "mcq": result["mcqs"],
            "one_line": result["one_lines"],
            "session_id": session_id,
            "token_usage": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
                "estimated_cost": estimated_cost
            }
        })

    except Exception as e:
        logger.error(f"💥 System error: {str(e)}")
        update_progress(session_id, 0, f"System error: {str(e)}")
        return jsonify({
            "mcq": "", 
            "one_line": f"⚠️ System error: {str(e)}", 
            "session_id": session_id
        })

@app.route("/progress/<session_id>")
def get_progress(session_id):
    """Enhanced progress endpoint with detailed information"""
    progress_info = progress_data.get(session_id, {
        "progress": 0,
        "status": "unknown",
        "details": {}
    })
    
    # Add debug logging to see what's being sent
    logger.info(f"📊 Sending progress for {session_id}: {progress_info}")
    
    # Clean up old progress data (older than 1 hour)
    current_time = time.time()
    for sid, data in list(progress_data.items()):
        if current_time - data.get("timestamp", 0) > 3600:  # 1 hour
            del progress_data[sid]
    
    return jsonify(progress_info)

# Keep all other existing routes
@app.route("/evaluate_answers", methods=["POST"])
def evaluate_answers_endpoint():
    data = request.get_json() or {}
    mcqs = data.get("mcqs", [])
    one_lines = data.get("one_lines", [])
    
    total_questions = len(mcqs) + len(one_lines)
    logger.info(f"📝 Starting evaluation of {total_questions} questions (rate-limit safe mode)")
    
    if total_questions == 0:
        return jsonify({"error": "No questions to evaluate"})
    
    try:
        result = pdf_qa_service.evaluate_student_answers(mcqs, one_lines)
        
        if "error" in result:
            logger.error(f"Evaluation error: {result['error']}")
            return jsonify(result)
        
        logger.info("✅ Evaluation completed successfully")
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"💥 Evaluation system error: {str(e)}")
        return jsonify({"error": f"Evaluation failed: {str(e)}"})

@app.route("/more_qa", methods=["POST"])
def more_qa():
    data = request.get_json()
    user_prompt = data.get("prompt", "")
    profession = data.get("profession", "general")
    
    return jsonify({
        "mcq": "Additional MCQ questions would be generated here based on your request.",
        "one_line": "Additional short questions would be generated here based on your request."
    })

@app.route("/health")
def health_check():
    return jsonify({
        "status": "healthy",
        "cache_size": len(evaluation_cache),
        "active_sessions": len(progress_data),
        "timestamp": time.time()
    })

if __name__ == "__main__":
    import nest_asyncio
    nest_asyncio.apply()
    logger.info("🚀 Starting PDF Q&A Generator with Enhanced Progress Tracking...")
    logger.info("🔒 Rate limit handling: ENABLED")
    logger.info("⚡ Sequential evaluation: ENABLED") 
    logger.info("💾 Caching: ENABLED")
    logger.info("📊 Progress tracking: ENHANCED")
    logger.info("🔗 Server running at: http://localhost:5000")
    app.run(debug=True, threaded=True, host="0.0.0.0", port=5000)