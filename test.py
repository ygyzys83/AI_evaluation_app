import ollama
from models import JudgeVerdict

response = ollama.generate(
    model="gpt-oss:20b",
    prompt="QUESTION: Who won? GROUND TRUTH: The Lakers. AI ANSWER: Lakers won.",
    system="Return a JSON object with fields: verdict (PASS or FAIL), similarity_score (1-5), reasoning (one sentence).",
    format=JudgeVerdict.model_json_schema(),
    options={"temperature": 0.0},
)
print(repr(response["response"]))
