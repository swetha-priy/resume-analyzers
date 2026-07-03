import streamlit as st
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import PyPDF2
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from collections import Counter
import tempfile
import os
import io

try:
    import docx2txt
except ImportError:
    docx2txt = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

try:
    from fpdf import FPDF
except ImportError:
    FPDF = None

try:
    import language_tool_python
except ImportError:
    language_tool_python = None

for resource, path in [
    ("punkt", "tokenizers/punkt"),
    ("stopwords", "corpora/stopwords"),
    ("averaged_perceptron_tagger", "taggers/averaged_perceptron_tagger"),
]:
    try:
        nltk.data.find(path)
    except LookupError:
        try:
            nltk.download(resource, quiet=True)
        except Exception:
            pass

SKILL_TERMS = {
    "python", "java", "c++", "c#", "sql", "excel", "communication", "leadership",
    "project management", "machine learning", "deep learning", "nlp", "data analysis",
    "aws", "azure", "gcp", "docker", "kubernetes", "javascript", "react", "node.js",
    "pandas", "numpy", "tensorflow", "pytorch", "sql server", "postgresql", "mongodb",
    "git", "linux", "html", "css", "salesforce", "crm", "agile", "scrum",
    "data visualization", "tableau", "power bi", "jira", "communication skills", "teamwork",
    "problem solving", "presentation", "analytical", "planning", "risk management",
    "business analysis", "quality assurance", "ux", "ui", "design", "seo", "sem",
    "content marketing", "social media", "product management", "cloud", "security",
    "devops", "automation", "testing", "sdlc", "api", "microservices",
    "tensorflow", "keras", "matlab", "scala", "ruby", "swift", "go", "rust",
    "spark", "hadoop", "etl", "big data", "finance", "accounting", "compliance",
    "operations", "customer support", "sales", "marketing", "analytics", "research"
}

st.set_page_config(page_title="ATS Resume Analyzer", page_icon="📄", layout="wide")
st.title("ATS Resume Analyzer")

st.markdown(
    """
    Upload your resume (PDF or DOCX) and paste a job description to get an ATS-style score, skill matching, missing keywords, improvement suggestions, and a downloadable report.
    """
)

with st.sidebar:
    st.header("What this tool does")
    st.info(
        """
        - Computes an ATS-style resume score
        - Extracts skills from resume and job description
        - Finds missing keywords and role-specific terms
        - Uses transformer-based semantic matching when available
        - Generates a downloadable PDF report
        """
    )
    st.header("How it works")
    st.write(
        """
        1. Upload a PDF or DOCX resume
        2. Paste your target job description
        3. Click **Analyze Match**
        4. Review the score, skill gaps, and suggestions
        """
    )

if SentenceTransformer is None:
    st.sidebar.warning(
        "For semantic matching, install sentence-transformers (pip install sentence-transformers). The app will still run with TF-IDF fallback."
    )
if docx2txt is None:
    st.sidebar.warning(
        "DOCX support requires docx2txt (pip install docx2txt). PDF resumes still work without it."
    )
if FPDF is None:
    st.sidebar.warning(
        "Downloadable PDF reports require fpdf (pip install fpdf)."
    )
if language_tool_python is None:
    st.sidebar.warning(
        "Grammar analysis works best with language-tool-python installed (pip install language-tool-python). A basic fallback will still run."
    )


def extract_text_from_pdf(uploaded_file):
    try:
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        text = ""
        for page in pdf_reader.pages:
            page_text = page.extract_text() or ""
            text += page_text + "\n"
        return text.strip()
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return ""


def extract_text_from_docx(uploaded_file):
    if docx2txt is None:
        st.error("Please install docx2txt to upload DOCX resumes.")
        return ""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name
    try:
        text = docx2txt.process(tmp_path) or ""
        return text.strip()
    except Exception as e:
        st.error(f"Error reading DOCX: {e}")
        return ""
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def extract_resume_text(uploaded_file):
    filename = uploaded_file.name.lower()
    if filename.endswith(".pdf"):
        return extract_text_from_pdf(uploaded_file)
    if filename.endswith(".docx"):
        return extract_text_from_docx(uploaded_file)
    st.error("Unsupported file type. Please upload a PDF or DOCX resume.")
    return ""


def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\+\#\.\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def get_stopwords():
    try:
        return set(stopwords.words("english"))
    except LookupError:
        # Fallback stopword set for environments where NLTK data is unavailable
        return {
            "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
            "any", "are", "as", "at", "be", "because", "been", "before", "being", "below",
            "between", "both", "but", "by", "could", "did", "do", "does", "doing", "down",
            "during", "each", "few", "for", "from", "further", "had", "has", "have", "having",
            "he", "her", "here", "hers", "herself", "him", "himself", "his", "how", "i",
            "if", "in", "into", "is", "it", "its", "itself", "let", "me", "more", "most",
            "my", "myself", "nor", "of", "on", "once", "only", "or", "other", "ought", "our",
            "ours", "ourselves", "out", "over", "own", "same", "she", "should", "so",
            "some", "such", "than", "that", "the", "their", "theirs", "them", "themselves",
            "then", "there", "these", "they", "this", "those", "through", "to", "too",
            "under", "until", "up", "very", "was", "we", "were", "what", "when", "where",
            "which", "while", "who", "whom", "why", "with", "would", "you", "your", "yours",
            "yourself", "yourselves"
        }


def safe_word_tokenize(text):
    try:
        return word_tokenize(text)
    except LookupError:
        return re.findall(r"\b\w+\b", text)


def remove_stopwords(text):
    stop_words = get_stopwords()
    tokens = safe_word_tokenize(text)
    return " ".join([token for token in tokens if token.lower() not in stop_words])


@st.cache_resource
def load_sentence_model():
    if SentenceTransformer is None:
        return None
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def semantic_similarity_score(resume_text, job_description, model):
    if model is None:
        return None
    try:
        embeddings = model.encode([resume_text, job_description], convert_to_numpy=True, normalize_embeddings=True)
        score = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0] * 100
        return round(float(score), 2)
    except Exception:
        return None


def tfidf_similarity_score(resume_text, job_description):
    resume_processed = remove_stopwords(clean_text(resume_text))
    job_processed = remove_stopwords(clean_text(job_description))
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([resume_processed, job_processed])
    score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0] * 100
    return round(float(score), 2)


def extract_keywords(text, top_n=30):
    cleaned = clean_text(text)
    tokens = [token for token in word_tokenize(cleaned) if len(token) > 1]
    filtered = [token for token in tokens if token not in stopwords.words("english")]
    try:
        tagged = pos_tag(filtered)
        candidates = [token for token, pos in tagged if pos.startswith("NN") or pos.startswith("JJ") or pos.startswith("VB")]
    except LookupError:
        candidates = filtered
    freq = Counter(candidates)
    return [word for word, _ in freq.most_common(top_n)]


def extract_key_phrases(text, top_n=15):
    cleaned = clean_text(text)
    tokens = [token for token in word_tokenize(cleaned) if len(token) > 1]
    filtered = [token for token in tokens if token not in stopwords.words("english")]
    candidates = []
    for i in range(len(filtered)):
        candidates.append(filtered[i])
        if i + 1 < len(filtered):
            candidates.append(f"{filtered[i]} {filtered[i+1]}")
    freq = Counter(candidates)
    return [phrase for phrase, _ in freq.most_common(top_n)]


def render_keyword_badges(keywords, color="#0f9d58", text_color="#ffffff"):
    if not keywords:
        return "<span style=\"color:#888;\">None</span>"
    badge_html = ""
    for keyword in keywords:
        badge_html += f"<span style=\"display:inline-block;background:{color};color:{text_color};border-radius:12px;padding:4px 10px;margin:2px;font-size:0.9em;\">{keyword}</span>"
    return badge_html


def extract_skills(text):
    text_lower = clean_text(text)
    found_skills = set()
    for skill in SKILL_TERMS:
        pattern = r"\b" + re.escape(skill) + r"\b"
        if re.search(pattern, text_lower):
            found_skills.add(skill)
    extra_keywords = extract_keywords(text, top_n=50)
    for keyword in extra_keywords:
        if keyword in SKILL_TERMS:
            found_skills.add(keyword)
    return sorted(found_skills)


def match_skills(resume_skills, job_skills):
    resume_set = {skill.lower() for skill in resume_skills}
    job_set = {skill.lower() for skill in job_skills}
    matched = sorted([skill for skill in job_set if skill in resume_set])
    missing = sorted([skill for skill in job_set if skill not in resume_set])
    return matched, missing


def calculate_ats_score(semantic_score, matched_skills_count, total_job_skills):
    skill_ratio = matched_skills_count / total_job_skills if total_job_skills else 0
    semantic_weight = 0.65 if semantic_score is not None else 0.0
    keyword_weight = 0.35 if total_job_skills else 1.0
    base_semantic = semantic_score if semantic_score is not None else 0
    score = base_semantic * semantic_weight + skill_ratio * 100 * keyword_weight
    return round(score, 2)


def build_suggestions(semantic_score, ats_score, matched, missing, resume_text, job_description):
    suggestions = []
    if missing:
        suggestions.append("Add the missing job skills below to your resume if you have experience with them.")
        suggestions.append("Use the exact role-specific keywords from the job description in your summary and experience sections.")
    else:
        suggestions.append("Your resume already includes the main job keywords. Keep your achievements quantitative and concise.")

    if semantic_score is not None and semantic_score < 65:
        suggestions.append("Improve alignment by tailoring your resume bullets to the job's core responsibilities.")
    if ats_score < 60:
        suggestions.append("Highlight results with numbers, percentages, or business outcomes for stronger ATS and recruiter appeal.")
    if len(missing) > 3:
        suggestions.append("Reorder your skills section so the most relevant keywords appear first.")
    if "resume" in job_description.lower() or "application" in job_description.lower():
        suggestions.append("Mention your most relevant technical skills near the top of the document.")
    return suggestions


def simple_grammar_analysis(text):
    issues = []
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    long_sentences = [s for s in sentences if len(s.split()) > 35]
    if long_sentences:
        issues.append("One or more sentences are very long. Break them into shorter, more concise bullets.")
    repeated = re.findall(r"\b(\w+)\s+\1\b", text, flags=re.IGNORECASE)
    if repeated:
        issues.append("Repeated word detected: " + ", ".join(dict.fromkeys([w.lower() for w in repeated])))
    if re.search(r"\b(i|me|my|mine)\b", text, flags=re.IGNORECASE):
        issues.append("Personal pronouns are present. Use a professional, third-person tone for resumes.")
    if not issues:
        issues.append("No obvious grammar issues detected in the current resume text.")
    return issues


def grammar_analysis(text):
    if language_tool_python is not None:
        try:
            tool = language_tool_python.LanguageTool("en-US")
            matches = tool.check(text)
            if not matches:
                return ["No grammar issues detected."]
            issues = []
            for match in matches[:8]:
                text_context = match.context if hasattr(match, "context") else ""
                error_text = text[match.offset: match.offset + match.errorLength] if match.errorLength else ""
                issue = match.message
                if error_text:
                    issue += f" Example: '{error_text.strip()}'"
                elif text_context:
                    issue += f" Context: '{text_context.strip()}'"
                issues.append(issue)
            return issues
        except Exception:
            pass
    return simple_grammar_analysis(text)


def create_pdf_bytes(report_data):
    if FPDF is None:
        return None
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "ATS Resume Match Report", ln=True, align="C")
    pdf.ln(5)
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 8, f"ATS Score: {report_data['ats_score']}%", ln=True)
    pdf.cell(0, 8, f"Semantic Match: {report_data['semantic_score']}%", ln=True)
    pdf.cell(0, 8, f"Keyword Match: {report_data['keyword_coverage']}%", ln=True)
    pdf.ln(5)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "Matched Skills:", ln=True)
    pdf.set_font("Arial", size=11)
    pdf.multi_cell(0, 6, ", ".join(report_data["matched_skills"]) or "None")
    pdf.ln(2)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "Missing Job Skills:", ln=True)
    pdf.set_font("Arial", size=11)
    pdf.multi_cell(0, 6, ", ".join(report_data["missing_skills"]) or "None")
    pdf.ln(2)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, make_pdf_safe("Improvement Suggestions:"), ln=True)
    pdf.set_font("Arial", size=11)
    for suggestion in report_data["suggestions"]:
        pdf.multi_cell(0, 6, make_pdf_safe(f"- {suggestion}"))
    pdf.ln(4)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, make_pdf_safe("Resume Skills Extracted:"), ln=True)
    pdf.set_font("Arial", size=11)
    pdf.multi_cell(0, 6, make_pdf_safe(", ".join(report_data["resume_skills"]) or "None"))
    pdf.ln(4)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, make_pdf_safe("Job Description Skills Extracted:"), ln=True)
    pdf.set_font("Arial", size=11)
    pdf.multi_cell(0, 6, make_pdf_safe(", ".join(report_data["job_skills"]) or "None"))

    pdf_bytes = pdf.output(dest="S")
    return pdf_bytes.encode("latin-1") if isinstance(pdf_bytes, str) else pdf_bytes


def make_pdf_safe(text):
    if not isinstance(text, str):
        text = str(text)
    return text.encode("latin-1", errors="replace").decode("latin-1")


def display_chart(matched_count, missing_count):
    fig, ax = plt.subplots(figsize=(6, 3))
    categories = ["Matched Skills", "Missing Skills"]
    values = [matched_count, missing_count]
    colors = ["#0f9d58", "#ff4b4b"]
    ax.bar(categories, values, color=colors)
    ax.set_ylabel("Number of skills")
    ax.set_title("Matched vs Missing Job Skills")
    for index, value in enumerate(values):
        ax.text(index, value + 0.1, str(value), ha="center", va="bottom")
    st.pyplot(fig)


def main():
    uploaded_file = st.file_uploader("Upload your resume (PDF or DOCX)", type=["pdf", "docx"])
    job_description = st.text_area("Paste the job description", height=250)

    if "analysis_done" not in st.session_state:
        st.session_state["analysis_done"] = False

    analyze_clicked = st.button("Analyze Match")
    if analyze_clicked:
        if not uploaded_file:
            st.warning("Please upload your resume.")
        elif not job_description:
            st.warning("Please paste the job description.")
        else:
            with st.spinner("Analyzing your resume..."):
                resume_text = extract_resume_text(uploaded_file)
                if not resume_text:
                    st.error("Could not extract text from the resume. Please try another file.")
                else:
                    semantic_model = load_sentence_model()
                    semantic_score = semantic_similarity_score(resume_text, job_description, semantic_model)
                    tfidf_score = tfidf_similarity_score(resume_text, job_description)
                    similarity_label = "Transformer semantic score" if semantic_score is not None else "TF-IDF similarity score"
                    similarity_value = semantic_score if semantic_score is not None else tfidf_score

                    resume_skills = extract_skills(resume_text)
                    job_skills = extract_skills(job_description)
                    resume_keywords = extract_key_phrases(resume_text, top_n=12)
                    job_keywords = extract_key_phrases(job_description, top_n=12)
                    if not job_skills:
                        job_skills = extract_keywords(job_description, top_n=20)

                    matched_skills, missing_skills = match_skills(resume_skills, job_skills)
                    keyword_coverage = round((len(matched_skills) / len(job_skills) * 100), 2) if job_skills else 0
                    ats_score = calculate_ats_score(similarity_value, len(matched_skills), len(job_skills))
                    suggestions = build_suggestions(semantic_score, ats_score, matched_skills, missing_skills, resume_text, job_description)
                    grammar_issues = grammar_analysis(resume_text)

                    st.session_state["analysis_done"] = True
                    st.session_state["resume_text"] = resume_text
                    st.session_state["job_description"] = job_description
                    st.session_state["semantic_score"] = semantic_score
                    st.session_state["tfidf_score"] = tfidf_score
                    st.session_state["similarity_label"] = similarity_label
                    st.session_state["similarity_value"] = similarity_value
                    st.session_state["resume_skills"] = resume_skills
                    st.session_state["job_skills"] = job_skills
                    st.session_state["resume_keywords"] = resume_keywords
                    st.session_state["job_keywords"] = job_keywords
                    st.session_state["matched_skills"] = matched_skills
                    st.session_state["missing_skills"] = missing_skills
                    st.session_state["keyword_coverage"] = keyword_coverage
                    st.session_state["ats_score"] = ats_score
                    st.session_state["suggestions"] = suggestions
                    st.session_state["grammar_issues"] = grammar_issues

    if st.session_state.get("analysis_done"):
        resume_text = st.session_state.get("resume_text", "")
        job_description = st.session_state.get("job_description", "")
        semantic_score = st.session_state.get("semantic_score")
        similarity_label = st.session_state.get("similarity_label", "Similarity score")
        similarity_value = st.session_state.get("similarity_value", 0.0)
        resume_skills = st.session_state.get("resume_skills", [])
        job_skills = st.session_state.get("job_skills", [])
        resume_keywords = st.session_state.get("resume_keywords", [])
        job_keywords = st.session_state.get("job_keywords", [])
        matched_skills = st.session_state.get("matched_skills", [])
        missing_skills = st.session_state.get("missing_skills", [])
        keyword_coverage = st.session_state.get("keyword_coverage", 0.0)
        ats_score = st.session_state.get("ats_score", 0.0)
        suggestions = st.session_state.get("suggestions", [])
        grammar_issues = st.session_state.get("grammar_issues", [])

        st.subheader("Match Results")
        st.markdown("***")
        c1, c2 = st.columns(2)
        c1.markdown("**Resume keyword highlights**")
        c1.markdown(render_keyword_badges(resume_keywords, color="#0f9d58"), unsafe_allow_html=True)
        c2.markdown("**Job description highlights**")
        c2.markdown(render_keyword_badges(job_keywords, color="#1e88e5"), unsafe_allow_html=True)

        st.markdown("***")
        st.markdown("**Skill match summary**")
        sm1, sm2 = st.columns(2)
        sm1.markdown("**Matched job skills**")
        sm1.markdown(render_keyword_badges(matched_skills, color="#0f9d58"), unsafe_allow_html=True)
        sm2.markdown("**Missing job skills**")
        sm2.markdown(render_keyword_badges(missing_skills, color="#ff4b4b"), unsafe_allow_html=True)

        st.markdown("***")
        col1, col2, col3 = st.columns(3)
        col1.metric("ATS Score", f"{ats_score:.2f}%")
        col2.metric(similarity_label, f"{similarity_value:.2f}%")
        col3.metric("Keyword Coverage", f"{keyword_coverage:.2f}%")

        color_index = min(int(ats_score // 33), 2)
        gauge_colors = ["#ff4b4b", "#ffa726", "#0f9d58"]
        fig, ax = plt.subplots(figsize=(8, 0.5))
        ax.barh([0], [ats_score], color=gauge_colors[color_index])
        ax.set_xlim(0, 100)
        ax.set_yticks([])
        ax.set_xlabel("ATS Score")
        ax.set_title("Overall Resume Match Score")
        st.pyplot(fig)

        if ats_score < 40:
            st.warning("Low ATS score. Focus on adding the missing role-specific keywords and measurable achievements.")
        elif ats_score < 70:
            st.info("Good match. Fine-tune your skill section and tailor a stronger summary.")
        else:
            st.success("Strong match. Your resume is well aligned to this job description.")

        st.markdown("---")
        st.subheader("Skills and Keyword Analysis")
        st.write("**Matched skills:**", ", ".join(matched_skills) if matched_skills else "None found")
        st.write("**Missing skills from the job description:**", ", ".join(missing_skills) if missing_skills else "None found")
        st.write("**Resume skills extracted:**", ", ".join(resume_skills) if resume_skills else "None found")
        st.write("**Job description keywords extracted:**", ", ".join(job_skills) if job_skills else "None found")

        display_chart(len(matched_skills), len(missing_skills))

        st.markdown("---")
        st.subheader("Improvement Suggestions")
        for suggestion in suggestions:
            st.markdown(f"- {suggestion}")

        st.markdown("---")
        st.subheader("Grammar Analysis")
        if grammar_issues:
            for issue in grammar_issues:
                st.markdown(f"- {issue}")
        else:
            st.success("No grammar issues detected in the resume text.")

        if FPDF is not None:
            report_data = {
                "ats_score": ats_score,
                "semantic_score": similarity_value,
                "keyword_coverage": keyword_coverage,
                "matched_skills": [skill.title() for skill in matched_skills],
                "missing_skills": [skill.title() for skill in missing_skills],
                "suggestions": suggestions,
                "resume_skills": [skill.title() for skill in resume_skills],
                "job_skills": [skill.title() for skill in job_skills],
            }
            pdf_bytes = create_pdf_bytes(report_data)
            if pdf_bytes:
                st.download_button(
                    label="Download PDF Report",
                    data=pdf_bytes,
                    file_name="resume_match_report.pdf",
                    mime="application/pdf",
                )
        else:
            st.info("Install fpdf to download a PDF report: pip install fpdf")

if __name__ == "__main__":
    main()
