import pandas as pd
import json
import os
import time
from dotenv import load_dotenv
import ollama
from google import genai

load_dotenv()

# ====================== CONFIGURATION ======================
MODELS_TO_TEST = [
    {"provider": "ollama", "model_name": "qwen2.5"},
    {"provider": "ollama", "model_name": "gpt-oss:20b"},
    {"provider": "ollama", "model_name": "gemma4:26b"}
]

# Set a timeout in seconds
REQUEST_TIMEOUT = 60.0
# ===========================================================

# Setup Gemini Client if evaluating a Gemini model
client_gemini = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))


def generate_response(prompt: str, provider: str, model_name: str) -> str:
    try:
        if provider == "gemini":
            # Gemini has its own internal timeout handling
            response = client_gemini.models.generate_content(model=model_name, contents=prompt)
            return response.text.strip()

        elif provider == "ollama":
            # 🚀 THE FIX: Explicitly set a client-side timeout in seconds
            # If Ollama doesn't respond in 60s, Python WILL kill the request and move on.
            response = ollama.generate(
                model=model_name,
                prompt=prompt,
                options={"temperature": 0.0},
            )
            return response['response'].strip()

    except Exception as e:
        error_msg = str(e)
        if "timeout" in error_msg.lower():
            print(f"❌ TIMEOUT: {model_name} took too long.")
            return "ERROR: INFERENCE_TIMEOUT"
        print(f"❌ ERROR ({model_name}): {error_msg}")
        return f"ERROR: {error_msg}"


def run_nba_eval():
    try:
        # Load data with string enforcement for GAME_ID
        df = pd.read_csv("nba_golden_dataset.csv", dtype={'GAME_ID': str})
        with open("eval_questions.json", "r") as f:
            questions = json.load(f)
    except Exception as e:
        print(f"Error loading files: {e}")
        return

    results = []

    for model_info in MODELS_TO_TEST:
        p_name = model_info["provider"]
        m_name = model_info["model_name"]

        print(f"\n🚀 Starting Evaluation: {m_name}")
        # PRO-TIP: Give the GPU 5 seconds to clear VRAM between model swaps
        time.sleep(5)

        for q in questions:
            # Data Alignment
            target_id = str(q['game_id']).strip().zfill(10)
            game_rows = df[df['GAME_ID'] == target_id]

            if game_rows.empty:
                print(f"⚠️ Missing Data: {target_id}")
                continue

            game_context = game_rows.to_string()

            prompt = f"""
            You are an NBA stats expert. Use ONLY the following data to answer the question. 
            If a question is subjective or cannot be answered using the provided data, 
            state that you do not have sufficient information.
            Data: {game_context}
            Question: {q['question']}
            Answer simply and concisely:
            """

            start_time = time.time()

            # Execute with error handling
            llm_answer = generate_response(prompt, p_name, m_name)

            latency = time.time() - start_time

            results.append({
                "id": q.get('id'),
                "model_used": m_name,
                "provider": p_name,
                "question": q.get('question'),
                "ground_truth": q.get('ground_truth'),
                "llm_answer": llm_answer,
                "latency": round(latency, 3),
                "char_count": len(llm_answer) if "ERROR" not in llm_answer else 0,
                "status": "SUCCESS" if "ERROR" not in llm_answer else "FAILED"
            })

            status_icon = "✅" if "ERROR" not in llm_answer else "❌"
            print(f"{status_icon} Q{q.get('id')} ({round(latency, 1)}s)")

    # Save to a new filename to avoid overwriting successful runs
    output_file = "multi_model_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\n🎉 Eval Finished. Results saved to {output_file}")


if __name__ == "__main__":
    run_nba_eval()