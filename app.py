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
from groq import Groq


APP_TITLE = "Study Guide Tutor"
APP_DIR = os.path.dirname(os.path.abspath(__file__))
TRUSTED_URLS_FILE = os.path.join(APP_DIR, ".streamlit", "trusted_urls.json")
GROQ_VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"


def _safe_get_groq_api_key() -> Optional[str]:
    # Prefer Streamlit secrets, then environment variables.
    try:
        # Streamlit secrets behaves like a mapping; prefer direct access.
        if "GROQ_API_KEY" in st.secrets:  # type: ignore[operator]
            key = st.secrets["GROQ_API_KEY"]  # type: ignore[index]
            key = str(key).strip()
            if key and "PASTE_YOUR_KEY_HERE" not in key:
                return key
    except Exception:
        pass
    key = (os.getenv("GROQ_API_KEY") or "").strip()
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
        # If OneDrive/Windows permissions prevent writes, don't crash the app.
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
    "the",
    "a",
    "an",
    "and",
    "or",
    "to",
    "of",
    "in",
    "for",
    "on",
    "with",
    "at",
    "by",
    "from",
    "as",
    "is",
    "are",
    "was",
    "were",
    "be",
    "this",
    "that",
    "it",
    "you",
    "your",
    "they",
    "we",
    "i",
    "me",
    "my",
    "can",
    "will",
    "would",
    "should",
    "could",
    "do",
    "does",
    "did",
    "but",
    "not",
    "so",
    "if",
    "then",
    "than",
    "too",
    "very",
}


def _extract_keywords(question: str) -> List[str]:
    text = _normalize_text(question)
    tokens = re.findall(r"[a-z0-9]+", text)
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) >= 3]
    # de-duplicate but keep order
    seen = set()
    out = []
    for t in tokens:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out[:25]


def _split_into_paragraphs(text: str, max_paragraph_chars: int = 700) -> List[str]:
    # Keep it simple: split by blank lines; if a paragraph is huge, cut it.
    paras = re.split(r"\n\s*\n", text)
    cleaned: List[str] = []
    for p in paras:
        p = p.strip()
        if not p:
            continue
        # Strip excessive whitespace.
        p = re.sub(r"\s+", " ", p)
        while len(p) > max_paragraph_chars:
            cleaned.append(p[:max_paragraph_chars].strip())
            p = p[max_paragraph_chars:].strip()
        if p:
            cleaned.append(p)
    return cleaned[:60]


@st.cache_data(show_spinner=False)
def _fetch_and_extract_url_text(url: str) -> str:
    # Fetch HTML and extract main text.
    headers = {"User-Agent": "Mozilla/5.0 (compatible; study-tutor/1.0)"}
    resp = requests.get(url, headers=headers, timeout=25)
    resp.raise_for_status()
    downloaded = resp.text
    extracted = trafilatura.extract(downloaded, include_comments=False, include_tables=False)
    if not extracted:
        # Fallback: return the first part of the HTML-stripped content.
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
        # If we cannot find keywords, just take the first few paragraphs.
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

    combined = "\n\n".join(chosen)
    combined = combined[:max_chars]
    return combined


def _build_tutor_system_prompt(mode: str) -> str:
    base = (
        "You are a helpful tutor. Your job is to teach the student the method and reasoning. "
        "You MUST NOT give the final answer or complete solution (for example, a final number, final expression, "
        "or the last step that directly produces the result). "
        "Instead: provide hints, explain concepts, and guide them through the steps of the problem. "
        "When checking work, identify mistakes and explain how to correct them, but still do not provide the final result. "
        "Be concise and structured."
        "If the student provides an image of their work, use it to help them understand the problem and guide them through the steps."
        "If the student asks a question to solve an arthimetic problem, and if arthimetic isn't the topic, give them the answre of the arthimetic problem."
        "Give the student the answer to a problem if the problem isn't the topic."
        "Don't ask the student questions, (remember, you are a tutor, not a student), but guide them through the steps of the problem."
        "Make sure the student understands the problem and the steps to solve it."
        "If they are finished working with you, tell them what you need to review so they can master the topic."
        "Make sure you are apporpriate to the student's age and knowledge level."
        "Make sure you are not too verbose or too concise."
        "Make sure you are not too confusing or too simple."
        "Make sure you are not too boring or too engaging."
        "Make sure you are not too long or too short."
        "Make sure you are not too complex or too simple."
        "Make sure you are not too confusing or too simple."
        "Make sure you are not too boring or too engaging."
        "Make sure instead of asking questions of what they need to improve, just directly tell them what they need to improve."
        "Don't be repetitive, but don't be too concise either."
        "If they are leaving the chat, tell them have a good day and see you next time."
        "Follow all school guidelines and policies."
        "Don't do illegal or unethical things."
        "Don't do anything that is not appropriate for a school setting."
        "Don't do anything that is not appropriate for a student's age and knowledge level."
        "Don't do anything that is not appropriate for a student's school."
        "Don't do anything that is not appropriate for a student's school district."
        "Don't do anything that is not appropriate for a student's school state."
        "Don't do anything that is not appropriate for a student's school country."
        "Don't do anything that is not appropriate for a student's school city."
        "Also, if there are multiple steps to a problem and the user asks for the answer for a single step (not the whole problem) give them the answer to the single step."
        "Only give the answer to a step of the problem if the user asks. for example if the step of the problem is to import the random module in python, don't give them the answer unless the user explicitly asks for the answer to that step."
        "If the question isn't academic related just straight up give them an answer no hints no questions just the answer."
    )
    if mode == "teach":
        return (
            base
            + "\n\n (if school related) Teach mode: Explain the concept and outline the approach. Give them the structure of the problem and the steps to solve it (Ex. for area: l * w or for volume: l * w * h). End by prompting the student to attempt the next step. (wihout final answer). if it is not school related just give the answer straight up wtih out any questions"
        )
    if mode == "check":
        return (
            base
            + "\n\n (if school related) Check mode: Review the student's work. Tell them if they are correct or incorrect and how to correct it. Do not provide the correct final answer, but give them the answer to the problem if the problem isn't the topic. (without final answer). if it is not school related just straight up give the answer."
        )
    return (
        base
        + "\n\nBoth mode: First Teach mode: Explain the concept and outline the approach. Give them the structure of the problem and the steps to solve it (Ex. for area: l * w or for volume: l * w * h). End by prompting the student to attempt the next step. (without final answer)"
        "Then Review the student's work. Tell them if they are correct or incorrect and how to correct it. Do not provide the correct final answer, but give them the answer to the problem if the problem isn't the topic. (without final answer)"
    )


def _image_to_data_url(upload) -> str:
    # Streamlit's UploadedFile supports reading bytes.
    raw = upload.getvalue()
    # Best-effort mime type.
    mime = upload.type or "image/png"
    b64 = base64.b64encode(raw).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def _call_groq_tutor(
    client: Groq,
    *,
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
        "Output requirements:\n"
        "- Provide step-by-step hints.\n"
        "- Ask 1-3 short questions the student can answer.\n"
        "- Do NOT give the final answer/result."
    )

    content_parts = [{"type": "text", "text": user_text}]

    if image_upload is not None:
        data_url = _image_to_data_url(image_upload)
        content_parts.append(
            {
                "type": "image_url",
                "image_url": {"url": data_url},
            }
        )

    # Build full conversation: system prompt + prior turns (text only) + current user message (optional image).
    history_messages = history_messages[-16:]  # cap context for cost/latency
    api_messages = [
        {"role": "system", "content": system_prompt},
        *[
            {"role": m.get("role", "user"), "content": m.get("content", "")}
            for m in history_messages
            if m.get("role") in {"user", "assistant"}
        ],
        {"role": "user", "content": content_parts if image_upload is not None else user_text},
    ]

    # Groq vision models support `content` as a list of parts (text + image_url).
    resp = client.chat.completions.create(
        model=GROQ_VISION_MODEL,
        messages=api_messages,
        temperature=0.2,
        max_completion_tokens=1200,
    )

    return (resp.choices[0].message.content or "").strip()


@dataclass
class TutorConfig:
    mode: str
    include_trusted: bool


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)

    api_key = _safe_get_groq_api_key()
    if not api_key:
        st.error("Missing `GROQ_API_KEY`. Add it to Streamlit secrets or your environment variables.")
        st.stop()

    if "trusted_urls" not in st.session_state:
        st.session_state.trusted_urls = _load_trusted_urls()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Layout: sidebar for trusted URLs, main for chat + upload.
    with st.sidebar:
        st.subheader("Trusted Articles (URLs)")
        st.caption("These are used as study context. The AI will still teach without giving final answers.")
        new_url = st.text_input("Add a URL", placeholder="https://...")
        if st.button("Add URL"):
            url = (new_url or "").strip()
            if url and url not in st.session_state.trusted_urls:
                st.session_state.trusted_urls.append(url)
                _save_trusted_urls(st.session_state.trusted_urls)
                st.success("URL added.")
            elif url in st.session_state.trusted_urls:
                st.info("That URL is already added.")
            else:
                st.warning("Please enter a valid URL.")

        if st.session_state.trusted_urls:
            for i, url in enumerate(list(st.session_state.trusted_urls)):
                cols = st.columns([3, 1])
                cols[0].write(url)
                if cols[1].button("Remove", key=f"remove_{i}"):
                    st.session_state.trusted_urls.pop(i)
                    _save_trusted_urls(st.session_state.trusted_urls)
                    st.rerun()

        st.divider()
        include_trusted = st.checkbox("Use trusted URLs as context", value=True)
        mode = st.radio("AI Mode", ["both", "teach", "check"], index=0, horizontal=True)
        st.caption("`both` = teach first, then check work.")

    config = TutorConfig(mode=mode, include_trusted=include_trusted)

    left_col, right_col = st.columns([2, 1], gap="large")

    with left_col:
        st.subheader("Chat")
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg.get("display_content", msg["content"]))

        question = st.chat_input("Ask for help (or paste your question).")
        student_attempt_text = st.text_area("What have you tried so far? (optional)", height=90)

        with st.expander("Drop Homework Image (optional)", expanded=False):
            uploaded = st.file_uploader(
                "Drop image here",
                type=["png", "jpg", "jpeg"],
                accept_multiple_files=False,
                key="homework_uploader_left",
            )
            if uploaded is not None:
                st.session_state.uploaded_image = uploaded
                st.success("Image added.")
                st.image(st.session_state.uploaded_image, use_column_width=True)

            if st.session_state.get("uploaded_image") is not None and st.button("Clear image", key="clear_image_left"):
                st.session_state.uploaded_image = None
                st.rerun()

        if question:
            client = Groq(api_key=api_key)
            question_text = question.strip()
            attempt_text = (student_attempt_text or "").strip()
            if attempt_text:
                user_history_content = (
                    f"{question_text}\n\nStudent attempt (typed):\n{attempt_text}"
                )
            else:
                user_history_content = question_text

            st.session_state.messages.append(
                {"role": "user", "content": user_history_content, "display_content": question_text}
            )
            with st.chat_message("user"):
                st.markdown(question_text)

            with st.chat_message("assistant"):
                with st.spinner("Tutoring..."):
                    # Build trusted context.
                    trusted_context = ""
                    if config.include_trusted and st.session_state.trusted_urls:
                        excerpts = []
                        for url in st.session_state.trusted_urls[:6]:
                            try:
                                article_text = _fetch_and_extract_url_text(url)
                                excerpt = _select_relevant_excerpts(article_text, question)
                                if excerpt:
                                    excerpts.append(f"Source: {url}\n{excerpt}")
                            except Exception:
                                # Skip failing URLs silently.
                                continue
                        trusted_context = "\n\n---\n\n".join(excerpts)

                    # Image handled in right column.
                    image_upload = st.session_state.get("uploaded_image")

                    mode_for_prompt = config.mode
                    answer = _call_groq_tutor(
                        client,
                        mode=mode_for_prompt,
                        history_messages=st.session_state.messages[:-1],
                        question=question,
                        trusted_context=trusted_context,
                        student_attempt_text=student_attempt_text,
                        image_upload=image_upload,
                    )

                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})

    with right_col:
        st.subheader("Homework Image")
        st.caption("Preview of the image you attached (optional).")

        if st.session_state.get("uploaded_image") is not None:
            st.image(st.session_state.uploaded_image, use_column_width=True)
            if st.button("Clear image", key="clear_image_right"):
                st.session_state.uploaded_image = None
                st.rerun()


if __name__ == "__main__":
    main()

