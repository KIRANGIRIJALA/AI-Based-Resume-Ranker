import streamlit as st
import fitz  # PyMuPDF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
import string
import re

nltk.download('stopwords')
STOPWORDS = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    tokens = text.split()
    tokens = [word for word in tokens if word not in STOPWORDS]
    return " ".join(tokens)

def extract_text_from_pdf(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def rank_resume(resume_text, jd_text):
    documents = [resume_text, jd_text]
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(documents)
    score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    return round(score * 100, 2)

# Streamlit UI
st.set_page_config(page_title="AI Resume Ranker", layout="centered")
st.title("📄 AI-Based Resume Ranker")
st.write("Upload your resume and paste a job description to see how well it matches.")

# Resume Upload
resume_file = st.file_uploader("Upload Resume (PDF format)", type=["pdf"])
jd_text = st.text_area("Paste Job Description here")

if st.button("🔍 Analyze") and resume_file and jd_text:
    with st.spinner("Reading resume..."):
        resume_text = extract_text_from_pdf(resume_file)
        resume_clean = clean_text(resume_text)
        jd_clean = clean_text(jd_text)
        match_score = rank_resume(resume_clean, jd_clean)

        st.success(f"✅ Resume Match Score: **{match_score}%**")
        if match_score >= 80:
            st.markdown("🟢 **Excellent Match!**")
        elif 60 <= match_score < 80:
            st.markdown("🟡 **Good Match. Could Improve.**")
        else:
            st.markdown("🔴 **Weak Match. Consider editing your resume.**")
