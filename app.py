import streamlit as st
import pypdf
import time
import base64
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ----------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------
st.set_page_config(
    page_title="Supermentor Technologies",
    page_icon="🏢",
    layout="wide"
)

# ----------------------------------------------------
# BACKGROUND IMAGE FUNCTION
# ----------------------------------------------------
def add_bg_from_url():

    st.markdown(
        f"""
        <style>

        .stApp {{
            background-image: url("https://images.unsplash.com/photo-1526374965328-7f61d4dc18c5");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}

        .main-container {{
            background: rgba(15, 23, 42, 0.88);
            padding: 30px;
            border-radius: 20px;
        }}

        h1, h2, h3 {{
            color: #38bdf8;
            text-align: center;
        }}

        p {{
            color: white;
        }}

        .stTextArea textarea {{
            border-radius: 10px;
            background-color: #1e293b;
            color: white;
        }}

        .stFileUploader {{
            background-color: rgba(30,41,59,0.8);
            padding: 15px;
            border-radius: 10px;
        }}

        .stButton>button {{
            background: linear-gradient(to right, #0ea5e9, #38bdf8);
            color: white;
            border-radius: 12px;
            height: 50px;
            width: 100%;
            font-size: 18px;
            border: none;
        }}

        .stButton>button:hover {{
            background: linear-gradient(to right, #0284c7, #0ea5e9);
        }}

        .result-box {{
            background-color: rgba(30,41,59,0.9);
            padding: 20px;
            border-radius: 15px;
            text-align: center;
            color: white;
            font-size: 24px;
            margin-top: 20px;
        }}

        .footer {{
            text-align: center;
            color: lightgray;
            margin-top: 40px;
            font-size: 15px;
        }}

        </style>
        """,
        unsafe_allow_html=True
    )

add_bg_from_url()

# ----------------------------------------------------
# SIDEBAR
# ----------------------------------------------------
st.sidebar.title("🏢 Supermentor Technologies")

st.sidebar.info(
    """
    ### Intelligent Resume Matching System

    This AI-powered recruitment platform analyzes resumes and compares them with job descriptions using NLP techniques and Machine Learning algorithms.

    ### Features
    ✅ Resume Analysis  
    ✅ Skill Detection  
    ✅ Match Score  
    ✅ Recommendation Engine  
    ✅ Analytics Visualization  
    """
)

st.sidebar.success("System Status: Active")

# ----------------------------------------------------
# MAIN CONTAINER
# ----------------------------------------------------
st.markdown('<div class="main-container">', unsafe_allow_html=True)

# ----------------------------------------------------
# HEADER
# ----------------------------------------------------
st.markdown(
    "<h1>🏢 Supermentor Technologies</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<h2>Intelligent Resume Matching System using NLP Techniques</h2>",
    unsafe_allow_html=True
)

st.markdown("---")

st.markdown(
    """
    <p style='text-align:center; font-size:18px;'>
    AI-powered recruitment dashboard for resume analysis and intelligent candidate matching.
    </p>
    """,
    unsafe_allow_html=True
)

# ----------------------------------------------------
# PDF TEXT EXTRACTION
# ----------------------------------------------------
def extract_text(file):

    reader = pypdf.PdfReader(file)

    text = ""

    for page in reader.pages:

        if page.extract_text():
            text += page.extract_text()

    return text

# ----------------------------------------------------
# SIMILARITY FUNCTION
# ----------------------------------------------------
def get_similarity(resume, jd):

    vectorizer = TfidfVectorizer()

    vectors = vectorizer.fit_transform([resume, jd])

    score = cosine_similarity(vectors[0], vectors[1])

    return score[0][0]

# ----------------------------------------------------
# SKILLS EXTRACTION
# ----------------------------------------------------
def extract_skills(text):

    skills_list = [
        "python",
        "java",
        "sql",
        "machine learning",
        "html",
        "css",
        "javascript",
        "nlp",
        "data analysis",
        "communication",
        "cloud computing",
        "aws"
    ]

    found_skills = []

    text = text.lower()

    for skill in skills_list:

        if skill in text:
            found_skills.append(skill)

    return found_skills

# ----------------------------------------------------
# INPUTS
# ----------------------------------------------------
resume = st.file_uploader(
    "📄 Upload Resume (PDF)",
    type="pdf"
)

if resume is not None:
    st.success("✅ Resume uploaded successfully!")

jd = st.text_area(
    "📝 Enter Job Description"
)

# ----------------------------------------------------
# BUTTON
# ----------------------------------------------------
if st.button("🚀 Analyze Resume"):

    if resume is not None and jd.strip() != "":

        with st.spinner("Analyzing resume using AI models..."):

            time.sleep(2)

            resume_text = extract_text(resume)

            score = get_similarity(resume_text, jd)

            skills = extract_skills(resume_text)

        # ----------------------------------------------------
        # SCORE BOX
        # ----------------------------------------------------
        st.markdown(
            f"""
            <div class="result-box">
            ✅ Match Score: <b>{round(score * 100, 2)}%</b>
            </div>
            """,
            unsafe_allow_html=True
        )

        # ----------------------------------------------------
        # PROGRESS BAR
        # ----------------------------------------------------
        st.progress(int(score * 100))

        # ----------------------------------------------------
        # SKILLS
        # ----------------------------------------------------
        st.subheader("🎯 Skills Identified")

        if skills:
            st.success(", ".join(skills))
        else:
            st.warning("No predefined skills identified")

        # ----------------------------------------------------
        # PIE CHART
        # ----------------------------------------------------
        if skills:

            fig, ax = plt.subplots()

            ax.pie(
                [1] * len(skills),
                labels=skills,
                autopct='%1.1f%%'
            )

            st.subheader("📊 Skills Distribution")

            st.pyplot(fig)

        # ----------------------------------------------------
        # RECOMMENDATION
        # ----------------------------------------------------
        st.subheader("📌 Recommendation")

        if score > 0.7:
            st.success("Highly Recommended Candidate")

        elif score > 0.4:
            st.warning("Moderately Suitable Candidate")

        else:
            st.error("Low Match for Job Role")

    else:
        st.error("⚠️ Please upload resume and enter job description.")

# ----------------------------------------------------
# FOOTER
# ----------------------------------------------------
st.markdown(
    """
    <div class="footer">
    Developed by Reya Javid | Supermentor Technologies | Internship Project 2026
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown('</div>', unsafe_allow_html=True)
