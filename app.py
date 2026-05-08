import streamlit as st
import pypdf
import time
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(
    page_title="Supermentor Resume Matcher",
    page_icon="📄",
    layout="centered"
)

# ---------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------
st.markdown("""
<style>

body {
    background-color: #0f172a;
}

.main {
    background: linear-gradient(to right, #1e293b, #0f172a);
    color: white;
}

h1, h2, h3 {
    text-align: center;
    color: #38bdf8;
}

.stButton>button {
    background-color: #38bdf8;
    color: white;
    border-radius: 10px;
    height: 50px;
    width: 100%;
    font-size: 18px;
    border: none;
}

.stButton>button:hover {
    background-color: #0ea5e9;
}

.result-box {
    background-color: #1e293b;
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    font-size: 22px;
    color: white;
    margin-top: 20px;
}

.footer {
    text-align: center;
    color: gray;
    margin-top: 40px;
    font-size: 14px;
}

</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# HEADER
# ---------------------------------------------------
st.markdown("<h1>🏢 Supermentor Technologies</h1>", unsafe_allow_html=True)

st.markdown(
    "<h3>Intelligent Resume Matching System using NLP Techniques</h3>",
    unsafe_allow_html=True
)

st.markdown("---")

st.write(
    "AI-powered recruitment platform for resume analysis and candidate matching."
)

# ---------------------------------------------------
# PDF TEXT EXTRACTION
# ---------------------------------------------------
def extract_text(file):
    reader = pypdf.PdfReader(file)
    text = ""

    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text()

    return text

# ---------------------------------------------------
# SIMILARITY SCORE
# ---------------------------------------------------
def get_similarity(resume, jd):

    vectorizer = TfidfVectorizer()

    vectors = vectorizer.fit_transform([resume, jd])

    score = cosine_similarity(vectors[0], vectors[1])

    return score[0][0]

# ---------------------------------------------------
# SKILLS EXTRACTION
# ---------------------------------------------------
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
        "communication"
    ]

    found_skills = []

    text = text.lower()

    for skill in skills_list:
        if skill in text:
            found_skills.append(skill)

    return found_skills

# ---------------------------------------------------
# INPUTS
# ---------------------------------------------------
resume = st.file_uploader("📄 Upload Resume", type="pdf")

jd = st.text_area("📝 Enter Job Description")

# ---------------------------------------------------
# BUTTON ACTION
# ---------------------------------------------------
if st.button("🚀 Analyze Resume"):

    if resume is not None and jd.strip() != "":

        with st.spinner("Analyzing Resume..."):

            time.sleep(2)

            resume_text = extract_text(resume)

            score = get_similarity(resume_text, jd)

            skills = extract_skills(resume_text)

        # ---------------------------------------------------
        # SCORE DISPLAY
        # ---------------------------------------------------
        st.markdown(
            f"""
            <div class="result-box">
                ✅ Match Score: <b>{round(score * 100, 2)}%</b>
            </div>
            """,
            unsafe_allow_html=True
        )

        # ---------------------------------------------------
        # PROGRESS BAR
        # ---------------------------------------------------
        st.progress(int(score * 100))

        # ---------------------------------------------------
        # SKILLS SECTION
        # ---------------------------------------------------
        st.subheader("🎯 Skills Identified")

        if skills:
            st.success(", ".join(skills))
        else:
            st.warning("No predefined skills found")

        # ---------------------------------------------------
        # PIE CHART
        # ---------------------------------------------------
        if skills:

            fig, ax = plt.subplots()

            ax.pie(
                [1] * len(skills),
                labels=skills,
                autopct='%1.1f%%'
            )

            st.subheader("📊 Skills Distribution")

            st.pyplot(fig)

        # ---------------------------------------------------
        # RECOMMENDATION
        # ---------------------------------------------------
        st.subheader("📌 Recommendation")

        if score > 0.7:
            st.success("Highly Recommended Candidate")

        elif score > 0.4:
            st.warning("Moderately Suitable Candidate")

        else:
            st.error("Low Match for Job Role")

    else:
        st.error("Please upload resume and enter job description.")

# ---------------------------------------------------
# FOOTER
# ---------------------------------------------------
st.markdown(
    """
    <div class="footer">
        Developed by Reya Javid | Internship Project 2026
    </div>
    """,
    unsafe_allow_html=True
)
