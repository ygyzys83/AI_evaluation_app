import streamlit as st
import pandas as pd
import json
import plotly.express as px

# --- PAGE SETUP ---
st.set_page_config(page_title="AI Model Benchmark: NBA Stats", layout="wide")

st.title("🏆 AI Model Selection Dashboard")
st.markdown("""
This dashboard evaluates three models across **5 Key Metrics** to determine the best choice for an NBA Fact-Checking product.
""")

st.sidebar.header("💰 Business Cost Projection")
monthly_volume = st.sidebar.slider("Estimated Monthly Queries", 1000, 1000000, 50000)

# Simulate Price per 1k queries
costs = {"qwen2.5": 0.002, "gpt-oss:20b": 0.005, "gemma4:26b": 0.008, "gemini-2.5-flash-lite": 0.01}

st.sidebar.subheader("Projected Monthly Cost")
for model, cost in costs.items():
    total = (monthly_volume / 1000) * cost
    st.sidebar.write(f"**{model}:** ${total:,.2f}")


# --- LOAD DATA ---
try:
    with open("final_comparison_report.json", "r") as f:
        data = json.load(f)
    df = pd.DataFrame(data)
except FileNotFoundError:
    st.error("❌ 'final_comparison_report.json' not found. Please run your Eval and Judge scripts first.")
    st.stop()

# --- CALCULATE TPM METRICS ---
# We simulate a "Cost" metric here (Price per 1k tokens in USD)
cost_map = {
    "qwen2.5": 0.00,        # Local = Free
    "gpt-oss:20b": 0.00,        # Local = Free
    "gemma4:26b": 0.00  # Local = Free
}
df['cost_sim'] = df['model_used'].map(cost_map)

# Aggregate Stats
model_stats = df.groupby('model_used').agg({
    'grade': lambda x: (x == 'PASS').sum() / max((x != 'ERROR').sum(), 1) * 100,
    'similarity_score': 'mean',
    'latency': 'mean',
    'char_count': 'mean',
    'cost_sim': 'sum'
}).reset_index()

# --- 1. THE LEADERBOARD ---
st.subheader("Model Performance Summary")
cols = st.columns(3)

for i, model_name in enumerate(model_stats['model_used']):
    row = model_stats[model_stats['model_used'] == model_name].iloc[0]
    with cols[i % 3]:
        st.info(f"**{model_name.upper()}**")
        st.metric("Accuracy", f"{row['grade']:.1f}%")
        st.metric("Avg Latency", f"{row['latency']:.2f}s")
        st.metric("Avg Similarity", f"{row['similarity_score']:.1f}/5")

st.divider()

# --- 2. TRADE-OFF ANALYSIS (VISUALS) ---
c1, c2 = st.columns(2)

with c1:
    st.subheader("Accuracy vs. Speed")
    # A TPM wants models in the 'Top Left' (High Accuracy, Low Latency)
    fig_scatter = px.scatter(
        model_stats,
        x="latency",
        y="grade",
        text="model_used",
        size="char_count",
        labels={"latency": "Latency (Seconds)", "grade": "Accuracy (%)"},
        title="Efficiency Matrix (Bubble size = Verbosity)"
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

with c2:
    st.subheader("Semantic Consistency")
    fig_bar = px.bar(
        model_stats,
        x="model_used",
        y="similarity_score",
        color="model_used",
        title="Average Similarity to Ground Truth (1-5)"
    )
    st.plotly_chart(fig_bar, use_container_width=True)

# --- 3. TPM INSIGHTS ---
st.divider()
st.subheader("🧠 TPM Decision Log")
best_acc = model_stats.sort_values(by='grade', ascending=False).iloc[0]

t1, t2 = st.columns([1, 2])
with t1:
    st.success(f"**Top Performer:** {best_acc['model_used']}")
    st.write(f"""
    - **Winning Factor:** {best_acc['grade']:.1f}% Accuracy.
    - **Trade-off:** Average response time of {best_acc['latency']:.2f}s.
    """)

with t2:
    st.warning("**Product Recommendation:**")
    st.write("""
    If this were for a **Live Play-by-Play** feature, I would prioritize the model with the lowest **latency**. 
    However, for a **Historical Fact-Checker**, accuracy is our North Star. 
    Based on the data, I recommend deploying the model above with a RAG-based context injection.
    """)

# --- 4. RAW DATA EXPLORER ---
with st.expander("🔍 View Direct Side-by-Side Comparison"):
    q_id = st.selectbox("Filter by Question ID", options=sorted(df['id'].unique()))
    comparison = df[df['id'] == q_id][['model_used', 'llm_answer', 'grade', 'similarity_score', 'reasoning']]
    st.table(comparison)