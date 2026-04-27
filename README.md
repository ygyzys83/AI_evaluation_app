[![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Ollama](https://img.shields.io/badge/Ollama-000000?style=for-the-badge&logo=ollama&logoColor=white)](https://ollama.com/)
[![RAG](https://img.shields.io/badge/RAG-Retrieval--Augmented%20Generation-8A2BE2?style=for-the-badge)](https://en.wikipedia.org/wiki/Retrieval-augmented_generation)
[![nba_api](https://img.shields.io/badge/Data%20Source-nba_api-orange?style=for-the-badge)](https://github.com/swar/nba_api)


***🏀 NBA AI Evaluation & Model Selection Framework***



***📋 Executive Summary***

In the shift from deterministic to probabilistic software, "vibes-based" testing is a liability. This project demonstrates a production-grade Evaluation (Evals) Framework used to benchmark multiple Large Language Models (LLMs) against a "Source of Truth" NBA dataset.
By building a custom RAG (Retrieval-Augmented Generation) pipeline, I measured the trade-offs between Accuracy, Latency, and Cost across three local models and one cloud API to make a data-driven model selection for a sports-analytics product.

***🛠 Skills Demonstrated***

AI Evaluation (Evals): Designing a "Golden Dataset" and automated scoring rubric.

LLM-as-a-Judge: Implementing a secondary LLM to automate the QA of unstructured natural language outputs.

Model Selection & Benchmarking: Quantifying the performance of Qwen, Gemma, and Gemini models.

Data Integrity: Handling complex real-world data issues (Leading zero Game IDs, VRAM constraints).

AI Safety & Guardrails: Implementing instruction-following checks to prevent hallucinations and subjective bias.

***🏗 System Architecture***

The diagram below illustrates the end-to-end flow from official data retrieval to the final TPM decision dashboard.
```mermaid
graph TD
    subgraph "Data Acquisition"
    A[nba_api] -->|Fetch Box Scores| B[nba_golden_dataset.csv]
    C[Human Experts] -->|Define Questions| D[eval_questions.json]
    end

    subgraph "Inference Pipeline (RAG)"
    B -->|Context Injection| E[Python Test Runner]
    D -->|Queries| E
    E -->|Ollama| F[Local LLMs: Qwen/Gemma]
    E -->|Google AI| G[Gemini 2.5 Flash]
    end

    subgraph "Evaluation Framework"
    F & G -->|Raw Answers| H[multi_model_results.json]
    H --> I[LLM-as-a-Judge Scorer]
    I -->|Grading Pass/Fail| J[final_comparison_report.json]
    end

    subgraph "Product Leadership"
    J --> K[Streamlit Dashboard]
    K --> L[Model Selection Decision]
    end
   ```

***📊 Key Performance Indicators (KPIs)***

To provide a 360-degree view of model effectiveness, the system tracks five primary metrics:
Accuracy (Pass Rate): Binary metric determined by the "Judge" LLM comparing answers to Ground Truth.
Semantic Similarity: A 1-5 score measuring how closely the model's phrasing aligns with human-written truth.
Inference Latency: Measurement of the "Time-to-Answer" to ensure real-time product feasibility.
Cost of Correctness: A simulation of monthly API spend vs. accuracy at scale (e.g., 100k queries/month).
Instruction Following (Guardrails): Testing the model's ability to refuse subjective questions (e.g., "Who is the GOAT?") when told to stay strictly within the provided data.

***🚀 Technical Challenges & Solutions***

The "Leading Zero" Data Integrity Issue:

Problem: NBA Game IDs (e.g., 0042500101) were being truncated to integers (e.g., 42500101) by Pandas, breaking the RAG lookup.

Solution: Implemented explicit schema enforcement (dtype={'GAME_ID': str}) and string padding (zfill(10)) to ensure 100% retrieval accuracy.
Hardware-Constrained Inference

Problem: Large models (Gemma 26B) caused system hangs and VRAM exhaustion during bulk evaluation.

Solution: Built a Defensive Inference Wrapper with hard client-side timeouts (60s) and "GPU Cool-down" periods between model swaps to ensure pipeline reliability.

***📈 Dashboard Preview***

The final Streamlit Dashboard provides a Model Selection Matrix. It allows stakeholders to toggle between prioritizing Accuracy (for historical record-keeping) or Latency (for live play-by-play updates), providing a clear ROI for each model choice.

***💻 Setup***

Clone the repository.

Install dependencies: pip install -r requirements.txt.

Set up environment: Add GOOGLE_API_KEY to .env.

Run the pipeline:

nba_data_pull.py

run_evals.py

evaluate_results.py

streamlit run app.py
