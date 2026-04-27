import json
import ollama
import os

# CONFIGURATION: We'll use a reliable local model as our "Golden Judge"
JUDGE_MODEL = "gpt-oss:20b"


def grade_all_models():
    try:
        with open("multi_model_results.json", "r") as f:
            results = json.load(f)
    except FileNotFoundError:
        print("❌ Error: multi_model_results.json not found. Run run_evals.py first.")
        return

    final_graded_data = []

    print(f"⚖️ Starting Multi-Model Grading using {JUDGE_MODEL} as the Judge...")

    for entry in results:
        # Construct the Judge Prompt
        # We ask for TWO things: a Pass/Fail and a Similarity Score (1-5)
        judge_prompt = f"""
        You are an expert NBA stats auditor. Compare the AI's answer against the Ground Truth.

        QUESTION: {entry['question']}
        GROUND TRUTH: {entry['ground_truth']}
        AI'S ANSWER: {entry['llm_answer']}

        RULES:
        1. Accuracy: Does the AI provide the correct numbers/teams? (PASS/FAIL)
        2. Similarity: On a scale of 1-5, how semantically similar is the AI answer to the Truth?

        Return your response in this EXACT format:
        Verdict: [PASS or FAIL]
        Similarity: [1-5]
        """

        try:
            response = ollama.generate(model=JUDGE_MODEL, prompt=judge_prompt)
            output = response['response'].upper()

            # Simple parsing of the Judge's output
            grade = "PASS" if "VERDICT: PASS" in output else "FAIL"

            # Extract similarity score (default to 1 if parsing fails)
            similarity = 1
            for s in ["1", "2", "3", "4", "5"]:
                if f"SIMILARITY: {s}" in output:
                    similarity = int(s)

            # Update the entry with our new metrics
            entry['grade'] = grade
            entry['similarity_score'] = similarity

            final_graded_data.append(entry)
            print(f"Graded: {entry['model_used']} - Q{entry['id']}: {grade} (Sim: {similarity})")

        except Exception as e:
            print(f"❌ Error grading {entry['id']}: {e}")

    # Save the final consolidated report
    with open("final_comparison_report.json", "w") as f:
        json.dump(final_graded_data, f, indent=4)

    print("\n✅ All models graded! Final report saved to final_comparison_report.json")


if __name__ == "__main__":
    grade_all_models()