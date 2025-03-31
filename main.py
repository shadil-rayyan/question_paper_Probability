import streamlit as st
import chromadb
import fitz  # PyMuPDF
import re
import hashlib
import logging
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

# Initialize ChromaDB
chroma_client = chromadb.PersistentClient(path="./chroma_db")
embedding_fn = SentenceTransformerEmbeddingFunction("all-MiniLM-L6-v2")

# Configure logging
logging.basicConfig(
    filename='app.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def sanitize_name(name):
    """Ensure collection name compatibility"""
    return re.sub(r'[^a-zA-Z0-9_-]', '_', name)[:63]

def extract_questions(text):
    """Improved question extraction with various number formats"""
    questions = []
    seen = set()
    
    patterns = [
        r'\n\s*(\d+[\.\)])\s+',
        r'\n\s*([A-Z][\.\)])\s+',
        r'\n\s*\((\d+)\)\s+',
        r'\n\s*Question\s+(\d+)\s*:'
    ]
    
    for pattern in patterns:
        raw_questions = re.split(pattern, text)
        for i in range(1, len(raw_questions), 2):
            q_num = raw_questions[i]
            q_text = raw_questions[i+1].split('\n')[0].strip()
            if not q_text:
                continue
                
            q_hash = hashlib.md5(f"{q_num}{q_text}".encode()).hexdigest()
            
            if q_hash not in seen:
                seen.add(q_hash)
                questions.append({
                    "text": q_text,
                    "number": q_num,
                    "hash": q_hash
                })
    return questions

def calculate_probability(years_present, total_years):
    """Accurate probability calculation"""
    return min(100, round((len(years_present) / total_years * 100))) if total_years else 0

def generate_enhanced_report(analysis_results):
    """Structured HTML report with proper white background"""
    html = """<html><head><style>
        body { 
            background-color: #ffffff !important;
            color: #000000;
            font-family: Arial, sans-serif;
            margin: 2rem;
        }
        .subject { 
            margin-bottom: 2rem; 
            padding: 1rem;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .question { 
            margin: 1rem 0; 
            padding: 1rem; 
            border-left: 4px solid #007bff;
            background-color: #f8f9fa;
            border-radius: 4px;
        }
        .probability { 
            color: #2c3e50; 
            font-weight: bold;
            font-size: 1.1rem;
        }
        .year-list { 
            color: #6c757d;
            font-size: 0.9rem;
        }
        .summary { 
            padding: 1rem; 
            margin-bottom: 2rem;
            background-color: #e9ecef;
            border-radius: 8px;
        }
    </style></head><body>"""
    
    total_questions = len(analysis_results)
    years_analyzed = {y for res in analysis_results for y in res['years']}
    
    html += f"""<div class="summary">
        <h2>Analysis Summary</h2>
        <p>Total Unique Questions: {total_questions}</p>
        <p>Years Analyzed: {", ".join(sorted(years_analyzed))}</p>
    </div>"""
    
    for question in sorted(analysis_results, key=lambda x: (-int(x['probability'].replace('%', '')), x['text'])):
        html += f"""<div class="question">
            <p class="probability">{question['probability']} Recurrence Probability</p>
            <p>{question['text']}</p>
            <p class="year-list">Appeared in: {', '.join(sorted(question['years']))}</p>
        </div>"""
    
    html += "</body></html>"
    return html

# Streamlit Interface
st.set_page_config(page_title="Exam Predictor", layout="wide")
st.title("📚 Smart Exam Predictor")

# Session state management
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []

# Sidebar Configuration
with st.sidebar:
    st.header("⚙️ Settings")
    
    # AI Provider Selection
    llm_provider = st.selectbox(
        "AI Provider",
        ["Ollama", "Groq", "Gemini"],
        index=0,
        help="Select the AI provider for enhanced analysis"
    )
    
    # API Key Inputs
    if llm_provider == "Groq":
        groq_api_key = st.text_input("Groq API Key", type="password")
    elif llm_provider == "Gemini":
        gemini_api_key = st.text_input("Gemini API Key", type="password")
    
    # Subject Configuration
    subject_name = st.text_input("Enter Subject Name", help="Enter the subject name for all uploaded papers")
    
    # Analysis Parameters
    st.subheader("Analysis Settings")
    similarity_threshold = st.slider(
        "Similarity Threshold", 
        min_value=0.5, 
        max_value=1.0, 
        value=0.75,
        help="Threshold for considering questions similar"
    )
    min_question_length = st.number_input(
        "Minimum Question Length", 
        min_value=10, 
        value=25,
        help="Minimum characters to consider as valid question"
    )

    # Clear button
    if st.button("Clear All Uploads"):
        st.session_state.uploaded_files = []
        try:
            chroma_client.delete_collection("question_analysis")
        except Exception as e:
            logger.warning(f"Error clearing collection: {str(e)}")
        st.rerun()  # Changed from st.experimental_rerun()

# Main Interface
uploaded_files = st.file_uploader(
    "Upload Past Question Papers (PDF)", 
    type="pdf", 
    accept_multiple_files=True,
    help="Upload multiple PDFs from same subject for analysis"
)

if uploaded_files:
    st.session_state.uploaded_files = uploaded_files
    all_questions = []
    year_pattern = re.compile(r'(20\d{2})')
    processed_files = 0
    error_files = []

    # Initialize ChromaDB collection
    try:
        collection = chroma_client.get_or_create_collection(
            name="question_analysis",
            embedding_function=embedding_fn
        )
    except Exception as e:
        logger.error(f"Failed to initialize ChromaDB collection: {str(e)}")
        st.error("Failed to initialize database. Please try refreshing the page.")
        st.stop()

    # Process each file individually
    for file in uploaded_files:
        try:
            file_content = file.read()
            if not file_content:
                logger.warning(f"Empty file detected: {file.name}")
                error_files.append(file.name)
                continue

            with fitz.open(stream=file_content, filetype="pdf") as doc:
                text = " ".join([page.get_text() for page in doc])
                logger.info(f"Processing file: {file.name}")
                
                # Extract year from filename
                year_match = year_pattern.search(file.name)
                year = year_match.group(1) if year_match else "Unknown"

                # Process questions
                questions = extract_questions(text)
                valid_questions = []
                for q in questions:
                    if len(q['text']) >= min_question_length:
                        q.update({
                            "year": year,
                            "subject": sanitize_name(subject_name) if subject_name else "General"
                        })
                        valid_questions.append(q)
                        logger.debug(f"Found question: {q['text'][:50]}...")
                
                if valid_questions:
                    all_questions.extend(valid_questions)
                    processed_files += 1

        except Exception as e:
            logger.error(f"Failed to process {file.name}: {str(e)}")
            error_files.append(file.name)
            continue

    # Show processing summary
    if error_files:
        st.warning(f"Successfully processed {processed_files}/{len(uploaded_files)} files. "
                 f"Could not process: {', '.join(error_files)}")

    # Add valid questions to ChromaDB
    if all_questions:
        try:
            existing_hashes = set(collection.get()['ids'])
            new_questions = [q for q in all_questions if q['hash'] not in existing_hashes]
            
            if new_questions:
                collection.add(
                    ids=[q['hash'] for q in new_questions],
                    documents=[q['text'] for q in new_questions],
                    metadatas=[{"year": q["year"]} for q in new_questions]
                )
                logger.info(f"Added {len(new_questions)} new questions to database")

            # Process results
            analysis_results = []
            seen_hashes = set()
            total_years = len({q['year'] for q in all_questions})
            
            for question in all_questions:
                if question['hash'] in seen_hashes:
                    continue
                    
                similar = collection.query(
                    query_texts=[question["text"]],
                    n_results=10,
                    include=["metadatas", "distances"]
                )
                
                # Filter by similarity threshold
                valid_years = [
                    m['year'] for m, d in zip(similar['metadatas'][0], similar['distances'][0])
                    if 1 - d > similarity_threshold
                ]
                years = set(valid_years)
                
                probability = calculate_probability(years, total_years)
                
                analysis_results.append({
                    "text": question["text"],
                    "years": years,
                    "probability": f"{probability}%",
                    "subject": question["subject"]
                })
                seen_hashes.add(question['hash'])

            # Display Results
            if analysis_results:
                st.subheader("Analysis Results")
                
                cols = st.columns(2)
                cols[0].metric("Total Unique Questions", len(analysis_results))
                cols[1].metric("Years Analyzed", len({y for res in analysis_results for y in res['years']}))
                
                with st.expander("View Detailed Report Preview"):
                    report = generate_enhanced_report(analysis_results)
                    st.components.v1.html(report, height=600, scrolling=True)
                
                st.download_button(
                    "📥 Download Full Report",
                    generate_enhanced_report(analysis_results),
                    "exam_analysis.html",
                    "text/html"
                )
            else:
                st.warning("No valid questions found in uploaded files")

        except Exception as e:
            logger.error(f"Database operation failed: {str(e)}")
            st.error("Failed to process questions. Please check the file formats and try again.")
    else:
        st.warning("No valid questions found in any uploaded files")