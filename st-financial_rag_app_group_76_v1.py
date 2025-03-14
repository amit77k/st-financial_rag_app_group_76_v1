import streamlit as st
import pdfplumber
import faiss
import numpy as np
import re
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, util
from thefuzz import process
from sklearn.preprocessing import MinMaxScaler
import nltk

# ✅ Ensure NLTK's Punkt tokenizer is available
nltk.download('punkt_tab')

from nltk.tokenize import sent_tokenize

# ✅ Load PDF
def load_pdf(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        text = "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])
    return text

# ✅ Extract Tables
def extract_tables_from_pdf(pdf_path):
    extracted_tables = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            tables = page.extract_tables()
            for table in tables:
                extracted_tables.append(table)
    return extracted_tables

# ✅ Chunk Text
def chunk_text(text, chunk_size=300):
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size//2)]

# ✅ Load Data
pdf_path = "BMW_Finance_NV_Annual_Report_2023.pdf"
pdf_text = load_pdf(pdf_path)
tables = extract_tables_from_pdf(pdf_path)
text_chunks = chunk_text(pdf_text)

# ✅ Embedding Model
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
chunk_embeddings = np.array([embedding_model.encode(chunk) for chunk in text_chunks])

# ✅ Initialize FAISS
dimension = chunk_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(chunk_embeddings)

# ✅ Initialize BM25
tokenized_chunks = [chunk.split() for chunk in text_chunks]
bm25 = BM25Okapi(tokenized_chunks)

# ✅ Improved Context Extraction (More Precise)
def extract_relevant_sentences(retrieved_chunks, query, max_sentences=6):
    sentences = []
    for chunk in retrieved_chunks:
        if not chunk or not chunk.strip():  # 🔹 Skip empty chunks
            continue
        chunk_sentences = sent_tokenize(chunk)  # ✅ Tokenize into sentences

        # ✅ Keep only sentences with financial data (numbers) or matching query terms
        for sentence in chunk_sentences:
            if re.search(r"\d{1,3}(?:[,.]\d{3})*(?:\.\d+)?", sentence) or any(word.lower() in sentence.lower() for word in query.split()):
                sentences.append(sentence)

    return " ".join(sentences[:max_sentences]) if sentences else "No relevant data found."
    
# ✅ Query Classification Fix (Better Threshold)
classification_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
relevant_keywords = [
    "revenue", "profit", "expenses", "income", "assets", "liabilities", "equity", 
    "earnings", "financial performance", "cash flow", "balance sheet", "receivables", 
    "accounts receivable", "trade receivables", "total receivables",
    "net loss", "operating expenses", "financial risk", "depreciation", "interest expense"
]

keyword_embeddings = classification_model.encode(relevant_keywords)
    
from thefuzz import process

def classify_query(query, threshold=0.45):  
    # First, check with embedding similarity
    query_embedding = classification_model.encode(query)
    similarity_scores = util.cos_sim(query_embedding, keyword_embeddings).squeeze().tolist()

    if similarity_scores and max(similarity_scores) >= threshold:
        return "relevant"
    
    # Fuzzy matching as a fallback
    best_match, score = process.extractOne(query, relevant_keywords)
    if score > 80:  # Adjust score threshold as needed
        return "relevant"

    return "irrelevant"

# ✅ Hallucination Filtering (Output-Side)
def filter_hallucinations(response, query, confidence_threshold=30):
    """
    Enhanced filter for hallucinated or misleading responses.
    """
    financial_keywords = [
        "trade receivables", "receivables", "affiliated companies", 
        "group companies", "bmw", "euro", "thousand", "total",
        "financial", "performance", "balance", "statement"
    ]
    
    # Check for presence of numbers and financial terms
    has_numbers = any(char.isdigit() for char in response)
    financial_terms_count = sum(1 for term in financial_keywords if term.lower() in response.lower())
    
    # Stricter confidence thresholds
    if confidence_threshold < 40 or not has_numbers:
        return "⚠️ Unable to provide a confident answer. Please verify with official financial statements."
    
    if financial_terms_count < 2:
        return "⚠️ The response may not contain sufficient financial context. Please verify with official documents."
    
    return response
    

def is_low_confidence_query(query):
    """
    Identify if a query is likely to be low confidence based on its characteristics
    """
    # List of vague terms that indicate low confidence queries
    vague_terms = [
        "how", "what about", "tell me about", "explain",
        "overview", "summary", "general", "trend",
        "performance", "doing well", "situation",
        "think", "feel", "believe", "around", "approximately",
        "roughly", "about", "changes", "difference"
    ]
    
    # Time-related vague terms
    vague_time_terms = [
        "past years", "recent", "lately", "over time",
        "historical", "history", "period", "timeline",
        "over the years", "previously"
    ]
    
    query_lower = query.lower()
    
    # Check for characteristics of low confidence queries
    has_vague_terms = any(term in query_lower for term in vague_terms)
    has_vague_time = any(term in query_lower for term in vague_time_terms)
    lacks_numbers = not any(char.isdigit() for char in query)
    is_short_query = len(query.split()) < 4
    
    # Count specific financial terms
    financial_terms = [
        "trade receivables", "receivables", "revenue", "profit",
        "balance sheet", "income statement", "cash flow",
        "assets", "liabilities", "equity", "earnings"
    ]
    specific_terms_count = sum(1 for term in financial_terms if term in query_lower)
    
    return {
        'is_low_confidence': (has_vague_terms or has_vague_time or 
                            (lacks_numbers and specific_terms_count == 0) or 
                            is_short_query),
        'reasons': {
            'has_vague_terms': has_vague_terms,
            'has_vague_time': has_vague_time,
            'lacks_numbers': lacks_numbers,
            'is_short_query': is_short_query,
            'specific_terms_count': specific_terms_count
        }
    }

# ✅ Multi-Stage Retrieval with Context Filtering , Hallucination Handling & Prompting
def multistage_retrieve(query, k=3, bm25_k=200, alpha=0.5):  # Adjusted alpha for better balance
    if not query or not query.strip():
        return "No query provided.", 0.0

    # Check for low confidence characteristics
    low_confidence_check = is_low_confidence_query(query)
    
    # Apply confidence penalty for low confidence queries
    confidence_penalty = 0.4 if low_confidence_check['is_low_confidence'] else 1.0
    
    # Enhance query preprocessing
    financial_terms = ["trade receivables", "receivables", "affiliated companies", 
                      "group companies", "total", "euro", "thousand", "financial", "bmw"]
    query_lower = query.lower()
    
    # Boost confidence if query contains specific financial terms
    term_matches = sum(1 for term in financial_terms if term in query_lower)
    term_boost = min(1.0, term_matches / 3)  # Normalize boost
    
    # Enhanced query prompt
    query_prompt = f"Provide a precise numerical answer for the following financial query, focusing on exact figures and dates: {query}"
    query_embedding = embedding_model.encode([query_prompt])
    
    # Improved BM25 scoring
    bm25_scores = bm25.get_scores(query.split())
    bm25_scores = np.array(bm25_scores)
    
    # Enhanced normalization with minimum threshold
    if len(bm25_scores) > 0:
        bm25_max = bm25_scores.max()
        if bm25_max > 0:
            bm25_scores = (bm25_scores / bm25_max) * 100
    
    # Get top BM25 matches
    top_bm25_indices = np.argsort(bm25_scores)[-bm25_k:]
    
    # Enhanced embedding similarity calculation
    filtered_embeddings = np.array([chunk_embeddings[i] for i in top_bm25_indices])
    faiss_index = faiss.IndexFlatIP(filtered_embeddings.shape[1])
    faiss_index.add(filtered_embeddings)
    
    # Get semantic search results
    _, faiss_ranks = faiss_index.search(query_embedding, k)
    top_faiss_indices = [top_bm25_indices[i] for i in faiss_ranks[0]]
    
    # Improved scoring combination
    final_scores = {}
    for i in set(top_bm25_indices) | set(top_faiss_indices):
        bm25_score = bm25_scores[i] if i in top_bm25_indices else 0
        faiss_score = float(np.dot(query_embedding, chunk_embeddings[i])) * 100
        
        # Combined score with term matching boost
        final_scores[i] = (alpha * bm25_score + (1 - alpha) * faiss_score) * (1 + term_boost)

    if final_scores:
        top_chunks = sorted(final_scores, key=final_scores.get, reverse=True)[:k]
        max_score = max(final_scores.values())
        retrieval_confidence = float(max_score * confidence_penalty)
        
        # Additional confidence adjustments
        if low_confidence_check['reasons']['has_vague_terms']:
            retrieval_confidence *= 0.7
        if low_confidence_check['reasons']['has_vague_time']:
            retrieval_confidence *= 0.8
        if low_confidence_check['reasons']['lacks_numbers']:
            retrieval_confidence *= 0.9
            
        retrieval_confidence = min(100, retrieval_confidence)
    else:
        top_chunks = []
        retrieval_confidence = 0.0

    valid_chunks = [i for i in top_chunks if i < len(text_chunks)]
    retrieved_chunks = [text_chunks[i] for i in valid_chunks] if valid_chunks else []
    
    # Enhanced context extraction
    precise_context = extract_relevant_sentences(retrieved_chunks, query)
    
    # Additional confidence boost for exact number matches
    if any(char.isdigit() for char in precise_context) and any(char.isdigit() for char in query):
        retrieval_confidence = min(100, retrieval_confidence * 1.2)

    # Apply hallucination filter
    final_response = filter_hallucinations(precise_context, query, retrieval_confidence)

    return final_response, round(retrieval_confidence, 2)

# ✅ Streamlit UI
st.title("📊 Financial Statement Q&A")
query = st.text_input("Enter your financial question:", key="financial_query")

if query:
    low_confidence_info = is_low_confidence_query(query)
    
    if low_confidence_info['is_low_confidence']:
        st.warning("⚠️ This appears to be a general or vague query. For better results, try to:")
        suggestions = []
        
        if low_confidence_info['reasons']['has_vague_terms']:
            suggestions.append("• Use specific financial metrics instead of general terms")
        if low_confidence_info['reasons']['has_vague_time']:
            suggestions.append("• Specify exact years or dates")
        if low_confidence_info['reasons']['lacks_numbers']:
            suggestions.append("• Include specific numerical references")
        if low_confidence_info['reasons']['is_short_query']:
            suggestions.append("• Provide more details in your query")
        
        for suggestion in suggestions:
            st.markdown(suggestion)
    
    retrieved_text, retrieval_confidence = multistage_retrieve(query)
    
    # Enhanced confidence score display with more detailed feedback
    if retrieval_confidence >= 80:
        st.success(f"### 🔍 Confidence Score: {retrieval_confidence}%\n\n"
                  f"✅ High Confidence Response:\n\n{retrieved_text}")
    elif retrieval_confidence >= 60:
        st.warning(f"### 🔍 Confidence Score: {retrieval_confidence}%\n\n"
                  f"⚠️ Medium Confidence Response:\n\n{retrieved_text}\n\n"
                  "*Please verify this information with official documents.*")
    else:
        st.error(f"### 🔍 Confidence Score: {retrieval_confidence}%\n\n"
                 f"❌ Low Confidence Response:\n\n{retrieved_text}\n\n"
                 "**Suggestion:** Try to:\n"
                 "- Be more specific in your question\n"
                 "- Include specific years or dates\n"
                 "- Ask about specific financial metrics\n"
                 "- Use terms from the financial statements")

        # Example of better query
        st.info("💡 **Example of a better query:**\n"
                '"What is the Trade receivables from BMW Group companies for year 2023?"')

# ✅ Testing & Validation
if st.sidebar.button("Run Test Queries"):
    st.sidebar.header("🔍 Testing & Validation")

    test_queries = [
        ("What is the Trade receivables from BMW Group companies for year 2023?", "High Confidence"),
        ("Can you tell me about the financial performance of BMW Group in 2023?", "Low Confidence"),
        ("What is the capital of France?", "Irrelevant")
    ]

    for test_query, confidence_level in test_queries:
        query_type = classify_query(test_query)

        if query_type == "irrelevant":
            st.sidebar.write(f"**🔹 Query:** {test_query} (❌ Irrelevant)")
            st.sidebar.write("**🔍 Confidence Score:** 0%")
            st.sidebar.write("⚠️ No relevant financial data available.")
            continue

        retrieved_text, retrieval_confidence = multistage_retrieve(test_query)
        st.sidebar.write(f"**🔹 Query:** {test_query}")
        st.sidebar.write(f"**🔍 Confidence Score:** {retrieval_confidence}%")
        #st.sidebar.success(f"✅ **Relevant Information:**\n\n {retrieved_text}")
        if retrieval_confidence >= 80:
            st.sidebar.success(f"✅ High Confidence\n\n **Relevant Context:**\n\n {retrieved_text}")
        else:
            st.sidebar.warning(f"⚠️ Low Confidence**\n\n **Relevant Context:** \n\n {retrieved_text}")
