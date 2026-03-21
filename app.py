import base64
import json
import os
import re
from dataclasses import dataclass
from io import BytesIO
from typing import List, Optional, Tuple

import requests
import streamlit as st
import trafilatura
import google.generativeai as genai


APP_TITLE = "Study Guide Tutor"
APP_DIR = os.path.dirname(os.path.abspath(__file__))
TRUSTED_URLS_FILE = os.path.join(APP_DIR, ".streamlit", "trusted_urls.json")


def _safe_get_gemini_api_key() -> Optional[str]:
    try:
        if "GEMINI_API_KEY" in st.secrets:
            key = str(st.secrets["GEMINI_API_KEY"]).strip()
            if key and "PASTE_YOUR_KEY_HERE" not in key:
                return key
    except Exception:
        pass

    key = (os.getenv("GEMINI_API_KEY") or "").strip()
    if key and "PASTE_YOUR_KEY_HERE" not in key:
        return key

    return None


def _load_trusted_urls() -> List[str]:
    if not os.path.exists(TRUSTED_URLS_FILE):
        return []
    try:
        with open(TRUSTED_URLS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        urls = data.get("urls", [])
        return [u for u in urls if isinstance(u, str) and u.strip()]
    except Exception:
        return []


def _save_trusted_urls(urls: List[str]) -> None:
    try:
        out_dir = os.path.dirname(TRUSTED_URLS_FILE)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(TRUSTED_URLS_FILE, "w", encoding="utf-8") as f:
            json.dump({"urls": urls}, f, indent=2)
    except PermissionError:
        st.warning(
            "Could not save `trusted_urls.json` (permission denied). "
            "The app will still work, but trusted URLs may not persist."
        )
    except Exception as e:
        st.warning(f"Could not save trusted URLs: {e}")


def _normalize_text(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s


STOPWORDS = {
    "the","a","an","and","or","to","of","in","for","on","with","at","by","from","as",
    "is","are","was","were","be","this","that","it","you","your","they","we","i","me","my",
    "can","will","would","should","could","do","does","did","but","not","so","if","then",
    "than","too","very",
}


def _extract_keywords(question: str) -> List[str]:
    text = _normalize_text(question)
    tokens = re.findall(r"[a-z0-9]+", text)
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) >= 3]
    seen = set()
    out = []
    for t in tokens:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out[:25]


def _split_into_paragraphs(text: str, max_paragraph_chars: int = 700) -> List[str]:
    paras = re.split(r"\n\s*\n", text)
    cleaned: List[str] = []
    for p in paras:
        p = p.strip()
        if not p:
            continue
        p = re.sub(r"\s+", " ", p)
        while len(p) > max_paragraph_chars:
            cleaned.append(p[:max_paragraph_chars].strip())
            p = p[max_paragraph_chars:].strip()
        if p:
            cleaned.append(p)
    return cleaned[:60]


@st.cache_data(show_spinner=False)
def _fetch_and_extract_url_text(url: str) -> str:
    headers = {"User-Agent": "Mozilla/5.0 (compatible; study-tutor/1.0)"}
    resp = requests.get(url, headers=headers, timeout=25)
    resp.raise_for_status()
    downloaded = resp.text
    extracted = trafilatura.extract(downloaded, include_comments=False, include_tables=False)
    if not extracted:
        extracted = trafilatura.extract(downloaded, output_format="txt") or ""
    return extracted.strip()[:60_000]


def _select_relevant_excerpts(article_text: str, question: str, max_chars: int = 2500) -> str:
    keywords = _extract_keywords(question)
    if not article_text:
        return ""
    paragraphs = _split_into_paragraphs(article_text)
    if not paragraphs:
        return ""

    if not keywords:
        chosen = paragraphs[:3]
    else:
        scored: List[Tuple[int, str]] = []
        for p in paragraphs:
            score = 0
            pl = p.lower()
            for k in keywords:
                if k in pl:
                    score += 1
            scored.append((score, p))
        scored.sort(key=lambda x: x[0], reverse=True)
        chosen = [p for score, p in scored if score > 0][:3] or paragraphs[:3]

    return "\n\n".join(chosen)[:max_chars]


def _build_tutor_system_prompt(mode: str) -> str:
    user_text = (
    f"Student question/work:\n{question.strip()}\n\n"
    f"Student attempt (typed):\n{student_attempt_text.strip() or '[none]'}\n\n"
    "Trusted materials (use these as supporting context when helpful):\n"
    f"{trusted_context.strip() or '[no trusted context available]'}\n\n"
    "Output requirements:\n"
    "- 1. Read the request to find what the problem is. If it is not a problem but an follow up question, the problem is the the problem before.\n"
    "- 2. If there is a problem, make sure you understand the topic and problem.\n"
    "- If the request is non-academic: answer directly (no hints and no Socratic questions).\n"
    "- If the request is academic: teach with steps/hints, and give the final answer while making sure they understand it. You can't just generate a essay or a report or just solve a math problem.\n"
    "- You make sure the user understands the problem, and then give the answer\n"
    "- Only ask questions if required to clarify missing info or which sub-step/part you should address.\n"
    "- If the user specifically asks for a sub-step/part, answer that sub-part directly (briefly)."
    "- Make sure all information is updated to 2026"
    "- If the user is asking you a follow up question without explicity saying which problem (like if they give you a math problem and then an ela essay and ask, Can you check it now?), use the most recent assignment (in this context it would be the essay becomes the essay comes after the math problem)"
    "- Don't display to user if the question is academic or not. keep it to yourself."
)

    return user_text


def _call_gemini_tutor(
    *,
    model,
    mode: str,
    history_messages: List[dict],
    question: str,
    trusted_context: str,
    student_attempt_text: str,
    image_upload,
) -> str:
    system_prompt = _build_tutor_system_prompt(mode)

    user_text = (
        f"Student question/work:\n{question.strip()}\n\n"
        f"Student attempt (typed):\n{student_attempt_text.strip() or '[none]'}\n\n"
        "Trusted materials (use these as supporting context when helpful):\n"
        f"{trusted_context.strip() or '[no trusted context available]'}\n\n"
    )

    history = [
        {"role": m["role"], "parts": [m["content"]]}
        for m in history_messages[-16:]
        if m.get("role") in {"user", "assistant"}
    ]

    chat = model.start_chat(history=history)

    if image_upload is not None:
        response = chat.send_message([
            system_prompt,
            user_text,
            {
                "mime_type": image_upload.type,
                "data": image_upload.getvalue()
            }
        ])
    else:
        response = chat.send_message(f"{system_prompt}\n\n{user_text}")

    return response.text.strip()


@dataclass
class TutorConfig:
    mode: str
    include_trusted: bool


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)

    api_key = _safe_get_gemini_api_key()
    if not api_key:
        st.error("Missing `GEMINI_API_KEY`.")
        st.stop()

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.0-flash")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    question = st.chat_input("Ask for help (or paste your question).")

    if question:
        answer = _call_gemini_tutor(
            model=model,
            mode="both",
            history_messages=st.session_state.messages,
            question=question,
            trusted_context="",
            student_attempt_text="",
            image_upload=None,
        )

        st.write(answer)


if __name__ == "__main__":
    main()