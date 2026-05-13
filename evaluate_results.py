import json
import ollama
from models import JudgeVerdict
from pydantic import ValidationError

# ── CONFIGURATION ─────────────────────────────────────────────────────────────
JUDGE_MODEL = "gpt-oss:20b"

JUDGE_SYSTEM_PROMPT = """You are an expert NBA stats auditor.
You will be given a question, a ground truth answer, and an AI's answer.
You must return a JSON object with exactly these fields:
- verdict: "PASS" or "FAIL" only
- similarity_score: integer from 1 to 5
- reasoning: one sentence explaining your verdict

GRADING RULES:
- PASS only if the AI's answer contains the correct facts from the ground truth
- If the question is subjective or out-of-scope, PASS if the AI correctly 
  refused to answer
- similarity_score 5 = identical meaning, 1 = completely wrong or unrelated
"""


def grade_entry(entry: dict) -> dict | None:
    """
    Grade a single eval entry using the judge model.
    Returns the entry with grade fields added, or None on unrecoverable failure.
    """
    judge_prompt = f"""
QUESTION: {entry['question']}
GROUND TRUTH: {entry['ground_truth']}
AI'S ANSWER: {entry['llm_answer']}
"""

    try:
        response = ollama.generate(
            model=JUDGE_MODEL,
            prompt=judge_prompt,
            system=JUDGE_SYSTEM_PROMPT,
            format=JudgeVerdict.model_json_schema(),  # forces JSON matching our schema
            options={"temperature": 0.0},
        )

        raw_json = response["response"]

        # Pydantic validates the structure — raises ValidationError if malformed
        verdict = JudgeVerdict.model_validate_json(raw_json)

        entry["grade"] = verdict.verdict
        entry["similarity_score"] = verdict.similarity_score
        entry["reasoning"] = verdict.reasoning
        return entry

    except ValidationError as e:
        print(f"  ⚠️  Validation failed for Q{entry['id']}: {e.error_count()} error(s)")
        entry["grade"] = "ERROR"
        entry["similarity_score"] = 0
        entry["reasoning"] = f"Validation error: {str(e)}"
        return entry

    except Exception as e:
        print(f"  ❌ Unexpected error for Q{entry['id']}: {e}")
        return None


def grade_all_models():
    try:
        with open("multi_model_results.json", "r") as f:
            results = json.load(f)
    except FileNotFoundError:
        print("❌ multi_model_results.json not found. Run run_evals.py first.")
        return

    print(f"⚖️  Grading with {JUDGE_MODEL} — structured output mode\n")

    final_graded_data = []
    errors = 0

    for entry in results:
        graded = grade_entry(entry)
        if graded:
            final_graded_data.append(graded)
            icon = "✅" if graded["grade"] == "PASS" else (
                   "⚠️ " if graded["grade"] == "ERROR" else "❌")
            print(
                f"{icon} {graded['model_used']} Q{graded['id']}: "
                f"{graded['grade']} (Sim: {graded['similarity_score']}) "
                f"— {graded['reasoning']}"
            )
        else:
            errors += 1

    with open("final_comparison_report.json", "w") as f:
        json.dump(final_graded_data, f, indent=4)

    print(f"\n✅ Done. {len(final_graded_data)} entries graded, {errors} dropped.")
    print("   Saved to final_comparison_report.json")


if __name__ == "__main__":
    grade_all_models()