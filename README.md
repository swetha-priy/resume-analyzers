# Resume Analyzer

A Streamlit-based ATS resume analyzer that scores resumes, highlights keyword match strength, and provides grammar feedback.

## Setup

1. Create a Python environment (recommended):
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

2. Install dependencies:
   ```powershell
   pip install -r requirements.txt
   ```

3. Download NLTK resources if needed (the app will also attempt this on first run):
   ```powershell
   python -m nltk.downloader punkt stopwords averaged_perceptron_tagger
   ```

## Run

```powershell
streamlit run app.py
```

## Notes

- `requirements.txt` includes optional packages used by the app: `docx2txt`, `sentence-transformers`, `fpdf`, and `language-tool-python`.
- If a dependency is missing, the app should still run with reduced functionality.
