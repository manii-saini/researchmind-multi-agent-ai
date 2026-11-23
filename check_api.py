from llm import backend_info, build_report, build_feedback, local_model

print("Checking local research backend (no API key required)...")
print("Backend:", backend_info())

if local_model():
    print("A local Ollama model is available for generated reports.")
else:
    print("No local model found; the built-in offline generator will be used.")

sample = build_report(
    "Keyless research pipeline",
    "Title: Local backend\nURL: https://example.com/local\nSnippet: The pipeline runs without an API key.",
)
print("Report generation:", "OK" if "Key Findings" in sample else "FAILED")
print("Critic generation:", "OK" if "Score:" in build_feedback(sample) else "FAILED")
