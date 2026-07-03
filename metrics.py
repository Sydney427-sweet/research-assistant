import json
import streamlit as st
import numpy as np

LOG_FILE = "logs/query_logs.json"

st.set_page_config(page_title="Metrics Dashboard", page_icon="📊", layout="wide")
st.title("📊 RAG Metrics Dashboard")

try:
    with open(LOG_FILE) as f:
        logs = json.load(f)
except FileNotFoundError:
    st.warning("No logs yet — ask some questions first!")
    st.stop()

if not logs:
    st.warning("No logs yet — ask some questions first!")
    st.stop()

# summary stats
total_queries = len(logs)
avg_total = np.mean([l["latency"]["total_ms"] for l in logs])
avg_retrieval = np.mean([l["latency"]["retrieval_ms"] for l in logs])
avg_rerank = np.mean([l["latency"]["rerank_ms"] for l in logs])
avg_generation = np.mean([l["latency"]["generation_ms"] for l in logs])

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Queries", total_queries)
col2.metric("Avg Total Latency", f"{avg_total:.0f}ms")
col3.metric("Avg Generation", f"{avg_generation:.0f}ms")
col4.metric("Avg Rerank", f"{avg_rerank:.0f}ms")

st.divider()

# latency breakdown chart
st.subheader("Latency per Query (ms)")
import pandas as pd
df = pd.DataFrame([{
    "query": f"Q{i+1}",
    "retrieval": l["latency"]["retrieval_ms"],
    "rerank": l["latency"]["rerank_ms"],
    "generation": l["latency"]["generation_ms"],
} for i, l in enumerate(logs)])
st.bar_chart(df.set_index("query"))

st.divider()

# query log table
st.subheader("Query Log")
for i, log in enumerate(reversed(logs)):
    with st.expander(f"Q{total_queries - i}: {log['question']}"):
        st.markdown(f"**Answer:** {log['answer']}")
        st.markdown(f"**Total latency:** {log['latency']['total_ms']:.0f}ms")
        st.markdown(f"**Reranker scores:** {log['reranker_scores']}")
        st.caption(log['timestamp'])
