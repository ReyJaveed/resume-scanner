import streamlit as st
import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------------
# Function to extract text from PDF
# -------------------------------
def extract_text(file):
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text()
    return text

# -------------------------------
# Function to calculate similarity
# -------------------------------
def get_similarity(resume, jd):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([resume, jd])
    score = cosine_similarity(vectors[0], vectors[1])
    return score[0][0]

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="Resume Screener", layout="centered")
st.title("AI Resume Screening System")
st.markdown("### 🚀 Smart AI Resume Analyzer")
st.markdown("---")
st.write("Upload your resume and compare it with a job description")

# File upload
resume = st.file_uploader("Upload Resume (PDF)", type="pdf")

# Job description input
jd = st.text_area("Enter Job Description")
def extract_skills(text):
    skills_list = ["python", "java", "sql", "machine learning", "data analysis", "html", "css"]
    found_skills = []

    text = text.lower()

    for skill in skills_list:
        if skill in text:
            found_skills.append(skill)

    return found_skills
# Button
if st.button("Check Match"):
    if resume is not None and jd.strip() != "":
        resume_text = extract_text(resume)

        score = get_similarity(resume_text, jd)
        skills = extract_skills(resume_text)

        st.success(f"Match Score: {round(score * 100, 2)}%")

        st.subheader("Skills Found in Resume:")
        st.write(skills if skills else "No predefined skills found")

    else:
        st.warning("Please upload a resume and enter job description")
