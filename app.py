import streamlit as st
import pypdf
import time

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(
    page_title="Supermentor Technologies",
    page_icon="🏢",
    layout="wide"
)

# ---------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------
st.markdown("""
<style>

/* MAIN APP */
.stApp {
    background: linear-gradient(to bottom right, #f8fbff, #eef4ff);
}

/* REMOVE DEFAULT PADDING */
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    padding-left: 2rem;
    padding-right: 2rem;
}

/* HEADINGS */
.main-title {
    font-size: 55px;
    font-weight: 800;
    color: #0f172a;
    margin-bottom: 0;
}

.sub-title {
    font-size: 34px;
    font-weight: 700;
    color: #2563eb;
    margin-top: 0;
}

.description {
    font-size: 20px;
    color: #475569;
    margin-top: 10px;
}

/* CARD */
.card {
    background-color: white;
    padding: 30px;
    border-radius: 25px;
    box-shadow: 0px 8px 25px rgba(0,0,0,0.08);
    margin-top: 30px;
}

/* BUTTON */
.stButton>button {
    background: linear-gradient(to right, #2563eb, #4f46e5);
    color: white;
    border: none;
    border-radius: 14px;
    height: 55px;
    width: 100%;
    font-size: 22px;
    font-weight: 600;
}

.stButton>button:hover {
    background: linear-gradient(to right, #1d4ed8, #4338ca);
    color: white;
}

/* TEXT AREA */
.stTextArea textarea {
    border-radius: 15px;
    border: 2px solid #dbeafe;
    font-size: 16px;
}

/* FILE UPLOADER */
.stFileUploader {
    border: 2px dashed #93c5fd;
    border-radius: 15px;
    padding: 15px;
    background-color: #f8fbff;
}

/* RESULT BOX */
.result-box {
    background: linear-gradient(to right, #2563eb, #4f46e5);
    padding: 25px;
    border-radius: 20px;
    text-align: center;
    color: white;
    font-size: 30px;
    font-weight: bold;
    margin-top: 25px;
}

/* FEATURE BOXES */
.feature-box {
    background-color: white;
    padding: 25px;
    border-radius: 18px;
    text-align: center;
    box-shadow: 0px 5px 15px rgba(0,0,0,0.05);
    height: 200px;
}

/* FOOTER */
.footer {
    text-align: center;
    margin-top: 50px;
    color: #64748b;
    font-size: 16px;
    font-weight: 500;
}

/* SIDEBAR */
section[data-testid="stSidebar"] {
    background: white;
    border-right: 1px solid #e2e8f0;
}

.sidebar-title {
    font-size: 34px;
    font-weight: 800;
    color: #0f172a;
}

.sidebar-subtitle {
    color: #2563eb;
    font-size: 18px;
    margin-bottom: 30px;
}

.sidebar-text {
    color: #334155;
    font-size: 16px;
    line-height: 1.7;
}

</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------
st.sidebar.markdown(
    """
    <div class="sidebar-title">
    🏢 Supermentor Technologies
    </div>

    <div class="sidebar-subtitle">
    AI For Smarter Hiring
    </div>
    """,
    unsafe_allow_html=True
)

st.sidebar.markdown("---")

st.sidebar.markdown(
    """
    <div class="sidebar-text">
    <b>ABOUT PROJECT</b><br><br>

    This AI-powered recruitment platform analyzes resumes and matches them with job descriptions using advanced NLP and Machine Learning techniques.
    </div>
    """,
    unsafe_allow_html=True
)

st.sidebar.markdown("---")

st.sidebar.success("✅ System Status: Active")

# ---------------------------------------------------
# HEADER
# ---------------------------------------------------
col1, col2 = st.columns([3,1])

with col1:

    st.markdown(
        """
        <div class="main-title">
        Supermentor Technologies
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="sub-title">
        AI-Powered Recruitment & Resume Matching System
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="description">
        Intelligent resume screening and candidate matching platform using Natural Language Processing and Machine Learning algorithms.
        </div>
        """,
        unsafe_allow_html=True
    )

with col2:
    st.image(
        "https://cdn-icons-png.flaticon.com/512/3135/3135715.png",
        width=180
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
# SIMILARITY FUNCTION
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
        "cloud computing",
        "aws",
        "communication",
        "data analysis"
    ]

    found_skills = []

    text = text.lower()

    for skill in skills_list:

        if skill in text:
            found_skills.append(skill)

    return found_skills

# ---------------------------------------------------
# MAIN CARD
# ---------------------------------------------------
st.markdown('<div class="card">', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:

    st.subheader("📄 Upload Resume (PDF)")

    resume = st.file_uploader(
        "",
        type="pdf"
    )

    if resume is not None:
        st.success("✅ Resume uploaded successfully!")

with col2:

    st.subheader("📝 Enter Job Description")

    jd = st.text_area(
        "",
        height=250,
        placeholder="Paste the job description here..."
    )

st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------------------------------
# ANALYZE BUTTON
# ---------------------------------------------------
if st.button("🚀 Analyze Resume"):

    if resume is not None and jd.strip() != "":

        with st.spinner("Analyzing Resume using AI..."):

            time.sleep(2)

            resume_text = extract_text(resume)

            score = get_similarity(resume_text, jd)

            skills = extract_skills(resume_text)

        # RESULT BOX
        st.markdown(
            f"""
            <div class="result-box">
            Match Score: {round(score * 100, 2)}%
            </div>
            """,
            unsafe_allow_html=True
        )

        # PROGRESS BAR
        st.progress(int(score * 100))

        st.markdown("<br>", unsafe_allow_html=True)

        # SKILLS
        st.subheader("🎯 Skills Identified")

        if skills:

            cols = st.columns(len(skills))

            for i, skill in enumerate(skills):

                cols[i].success(skill.upper())

        else:
            st.warning("No predefined skills found")

        # RECOMMENDATION
        st.subheader("📌 Recommendation")

        if score > 0.7:
            st.success("Highly Recommended Candidate")

        elif score > 0.4:
            st.warning("Moderately Suitable Candidate")

        else:
            st.error("Low Match for Job Role")

    else:
        st.error("Please upload resume and enter job description.")

st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------
# FEATURES SECTION
# ---------------------------------------------------
st.markdown("<br><br>", unsafe_allow_html=True)

st.markdown(
    """
    <h2 style='text-align:center; color:#0f172a;'>
    Why Choose Our AI Recruitment Platform?
    </h2>
    """,
    unsafe_allow_html=True
)

st.markdown("<br>", unsafe_allow_html=True)

f1, f2, f3, f4 = st.columns(4)

with f1:
    st.markdown(
        """
        <div class="feature-box">
        <h3>🧠 Advanced NLP</h3>
        <p>
        Uses Natural Language Processing for intelligent resume analysis.
        </p>
        </div>
        """,
        unsafe_allow_html=True
    )

with f2:
    st.markdown(
        """
        <div class="feature-box">
        <h3>🎯 Accurate Matching</h3>
        <p>
        Machine learning algorithms provide highly accurate matching scores.
        </p>
        </div>
        """,
        unsafe_allow_html=True
    )

with f3:
    st.markdown(
        """
        <div class="feature-box">
        <h3>📊 Analytics</h3>
        <p>
        Visual insights and analytics for better candidate evaluation.
        </p>
        </div>
        """,
        unsafe_allow_html=True
    )

with f4:
    st.markdown(
        """
        <div class="feature-box">
        <h3>🔒 Secure Platform</h3>
        <p>
        Secure and scalable AI-powered recruitment platform.
        </p>
        </div>
        """,
        unsafe_allow_html=True
    )

# ---------------------------------------------------
# FOOTER
# ---------------------------------------------------
st.markdown(
    """
    <div class="footer">
    © 2026 Supermentor Technologies | Developed by Reya Javid
    </div>
    """,
    unsafe_allow_html=True
)
