import os
from openai import OpenAI, RateLimitError
from langdetect import detect, LangDetectException

# Try to load OpenAI key (may be None)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None


def detect_language_offline(text: str) -> str:
    """Offline fallback using langdetect"""
    try:
        return detect(text).capitalize()
    except LangDetectException:
        return "Unknown"


def detect_language_openai(text: str) -> str:
    """
    Detect language using OpenAI.
    Falls back to offline detection if:
    - API key missing
    - Quota exceeded
    - Any OpenAI error occurs
    """

    # If no API key → go offline immediately
    if not client:
        return detect_language_offline(text) + " (offline)"

    try:
        prompt = (
            "Detect the language of the given text.\n"
            "Reply with ONLY the language name in English.\n\n"
            f"Text: {text}"
        )

        response = client.responses.create(
            model="gpt-5.2",
            input=prompt,
        )

        result = response.output_text.strip()
        return result if result else detect_language_offline(text)

    except RateLimitError:
        return detect_language_offline(text) + " (offline – quota exceeded)"

    except Exception:
        return detect_language_offline(text) + " (offline – API error)"
