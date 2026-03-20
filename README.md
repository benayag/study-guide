# study-guide
This is an app that helps you study for school work that might seem confusing. It uses AI to not give you the answer but nudge you to the right way to solve it.

## Run locally

1. Create a virtual environment and install dependencies:
   - `pip install -r requirements.txt`
2. Set your Groq key:
   - Streamlit secrets (recommended): put `GROQ_API_KEY="..."` in `.streamlit/secrets.toml`
   - Or via environment variable: `$env:GROQ_API_KEY="your_key_here"`
3. Start the app:
   - `streamlit run app.py`

## How it works

- Add URLs in the sidebar under `Trusted Articles (URLs)`.
- Ask a question and optionally upload an image of your homework.
- Choose `AI Mode`: `teach`, `check`, or `both`.
