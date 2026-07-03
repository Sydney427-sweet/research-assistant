import json
import os
from datetime import datetime

LOG_FILE = "logs/query_logs.json"

def init_logs():
    os.makedirs("logs", exist_ok=True)
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, "w") as f:
            json.dump([], f)

def log_query(question: str, answer: str, chunks: list, latency: dict):
    init_logs()

    entry = {
        "timestamp": datetime.now().isoformat(),
        "question": question,
        "answer": answer,
        "latency": {
            "retrieval_ms": round(latency.get("retrieval_ms", 0), 2),
            "rerank_ms": round(latency.get("rerank_ms", 0), 2),
            "generation_ms": round(latency.get("generation_ms", 0), 2),
            "total_ms": round(latency.get("total_ms", 0), 2),
        },
        "reranker_scores": latency.get("reranker_scores", []),
        "num_chunks": len(chunks),
    }

    with open(LOG_FILE, "r+") as f:
        logs = json.load(f)
        logs.append(entry)
        f.seek(0)
        json.dump(logs, f, indent=2)

    print(f"📝 Logged query | total={entry['latency']['total_ms']}ms")
