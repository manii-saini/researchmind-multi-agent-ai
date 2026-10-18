from pathlib import Path

from dotenv import dotenv_values
from langchain_openai import ChatOpenAI


env_file = Path(__file__).resolve().parent / ".env"
config = dotenv_values(env_file, encoding="utf-8-sig")
api_key = config.get("OPENAI_API_KEY")

if not api_key or not api_key.strip():
    print("ENV ERROR: OPENAI_API_KEY is missing.")
    raise SystemExit(1)

print("API key loaded. Testing OpenAI authentication...")

try:
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        api_key=api_key.strip(),
        temperature=0,
    )
    result = llm.invoke("Reply only with: WORKING")
    print("SUCCESS:", result.content)
except Exception as exc:
    print("API REQUEST FAILED:", type(exc).__name__)
    print("HTTP status:", getattr(exc, "status_code", None))
    print("API error code:", getattr(exc, "code", None))
    raise SystemExit(1) from None