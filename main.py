import streamlit as st
import chromadb
import fitz  # PyMuPDF
import re
import hashlib
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

# Initialize ChromaDB
chroma_client = chromadb.PersistentClient(path="./chroma_db")
embedding_fn = SentenceTransformerEmbeddingFunction("all-MiniLM-L6-v2")

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
    """Structured HTML report with categorization"""
    html = """<html><head><style>
        .subject { margin-bottom: 2rem; background: #f8f9fa; padding: 1rem; }
        .question { margin: 1rem 0; padding: 1rem; border-left: 4px solid #007bff; }
        .probability { color: #2c3e50; font-weight: bold; font-size: 1.1rem; }
        .year-list { color: #6c757d; font-size: 0.9rem; }
        .summary { background: #e9ecef; padding: 1rem; margin-bottom: 2rem; }
    </style></head><body>"""
    
    total_questions = len(analysis_results)
    subjects = {res['subject'] for res in analysis_results}
    years_analyzed = {y for res in analysis_results for y in res['years']}
    
    html += f"""<div class="summary">
        <h2>Analysis Summary</h2>
        <p>Total Unique Questions: {total_questions}</p>
        <p>Subjects Analyzed: {", ".join(subjects)}</p>
        <p>Years Covered: {", ".join(sorted(years_analyzed))}</p>
    </div>"""
    
    organized = {}
    for res in analysis_results:
        key = res['subject']
        organized.setdefault(key, []).append(res)
    
    for subject in sorted(organized.keys()):
        html += f"""<div class="subject">
            <h2>{subject}</h2>
            <p class="meta">Department: {organized[subject][0]['department']} | Semester: {organized[subject][0]['semester']}</p>"""
        
        for question in sorted(organized[subject], 
                             key=lambda x: int(x['probability'].replace('%', '')), 
                             reverse=True):
            html += f"""<div class="question">
                <p class="probability">{question['probability']} Recurrence Probability</p>
                <p>{question['text']}</p>
                <p class="year-list">Appeared in: {', '.join(sorted(question['years']))}</p>
            </div>"""
        
        html += "</div>"
    
    html += "</body></html>"
    return html

# Streamlit Interface
st.set_page_config(page_title="Academic Analyzer", layout="wide")
st.title("📚 Academic Question Analyzer")

# Sidebar Configuration
with st.sidebar:
    st.header("⚙️ Settings")
    
    # AI Provider Selection
    llm_provider = st.selectbox(
        "AI Provider",
        ["Ollama", "Groq", "Gemini"],
        index=0,
        help="Select the AI provider for metadata extraction"
    )
    
    # API Key Inputs
    if llm_provider == "Groq":
        groq_api_key = st.text_input("Groq API Key", type="password")
    elif llm_provider == "Gemini":
        gemini_api_key = st.text_input("Gemini API Key", type="password")
    
    # Default Metadata
    st.subheader("Default Metadata")
    default_department = st.text_input("Default Department")
    default_subject = st.text_input("Default Subject")
    default_semester = st.text_input("Default Semester")
    
    # Analysis Parameters
    st.subheader("Analysis Settings")
    similarity_threshold = st.slider(
        "Similarity Threshold", 
        min_value=0.5, 
        max_value=1.0, 
        value=0.7,
        help="Threshold for considering questions similar"
    )
    min_question_length = st.number_input(
        "Minimum Question Length", 
        min_value=10, 
        value=20,
        help="Minimum characters to consider as valid question"
    )

# Main Interface
uploaded_files = st.file_uploader(
    "Upload Question Papers (PDF)", 
    type="pdf", 
    accept_multiple_files=True
)

if uploaded_files:
    all_questions = []
    year_pattern = re.compile(r'(20\d{2})')
    metadata_patterns = {
        'department': r'Department:\s*([^\n]+)',
        'subject': r'Course (?:Code|Name):\s*([^\n]+)',
        'semester': r'Semester:\s*([^\n]+)'
    }

    for file in uploaded_files:
        with fitz.open(stream=file.read(), filetype="pdf") as doc:
            text = " ".join([page.get_text() for page in doc])
            year_match = year_pattern.search(file.name)
            year = year_match.group(1) if year_match else "Unknown"
            
            # Metadata extraction
            metadata = {}
            for key, pattern in metadata_patterns.items():
                match = re.search(pattern, text, re.IGNORECASE)
                default_value = locals().get(f"default_{key}", "Unknown")
                metadata[key] = sanitize_name(match.group(1).strip()) if match else default_value

            # Process questions
            questions = extract_questions(text)
            for q in questions:
                if len(q['text']) >= min_question_length:
                    q.update({
                        "year": year,
                        "department": metadata['department'],
                        "subject": metadata['subject'],
                        "semester": metadata['semester']
                    })
                    all_questions.append(q)

    # ChromaDB Analysis
    collection = chroma_client.get_or_create_collection(
        name="question_analysis",
        embedding_function=embedding_fn
    )
    
    existing_hashes = set(collection.get()['ids'])
    new_questions = [q for q in all_questions if q['hash'] not in existing_hashes]
    
    if new_questions:
        collection.add(
            ids=[q['hash'] for q in new_questions],
            documents=[q['text'] for q in new_questions],
            metadatas=[{"year": q["year"]} for q in new_questions]
        )
    
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
            "department": question["department"],
            "subject": question["subject"],
            "semester": question["semester"]
        })
        seen_hashes.add(question['hash'])
    
    # Display Results
    if analysis_results:
        st.subheader("Analysis Summary")
        cols = st.columns(3)
        cols[0].metric("Total Questions", len(analysis_results))
        cols[1].metric("Subjects Covered", len({res['subject'] for res in analysis_results}))
        cols[2].metric("Years Analyzed", len({y for res in analysis_results for y in res['years']}))
        
        with st.expander("View Detailed Report Preview"):
            report = generate_enhanced_report(analysis_results)
            st.components.v1.html(report, height=800, scrolling=True)
        
        st.download_button(
            "📥 Download Full Report",
            generate_enhanced_report(analysis_results),
            "question_analysis.html",
            "text/html"
        )
    else:
        st.warning("No valid questions found in uploaded files")