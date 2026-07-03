from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from test_questions import eval_dataset
from rag import load_vectorstore, build_rag_chain
import os


load_dotenv()

llm = ChatOllama(model="llama3.2", temperature=0)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def score_faithfulness(answer: str, chunks: list) -> float:
    """Ask the LLM if the answer is supported by the context."""
    context = "\n\n".join([c.page_content for c in chunks])
    prompt = f"""Given this context:
{context}

Is this answer fully supported by the context above? Answer only YES or NO.
Answer: {answer}"""
    response = llm.invoke(prompt).content.strip().upper()
    return 1.0 if "YES" in response else 0.0

def score_relevancy(question: str, answer: str) -> float:
    """Cosine similarity between question and answer embeddings."""
    q_vec = embeddings.embed_query(question)
    a_vec = embeddings.embed_query(answer)
    score = cosine_similarity([q_vec], [a_vec])[0][0]
    return round(float(score), 3)

def score_context_recall(question: str, chunks: list, ground_truth: str) -> float:
    """Ask LLM if the retrieved chunks contain enough info to answer correctly."""
    context = "\n\n".join([c.page_content for c in chunks])
    prompt = f"""Given this context:
{context}

Does the context contain enough information to produce this answer?
Answer: {ground_truth}

Reply only YES or NO."""
    response = llm.invoke(prompt).content.strip().upper()
    return 1.0 if "YES" in response else 0.0

def run_evaluation():
    print("🔧 Loading RAG pipeline...")
    vectorstore = load_vectorstore()
    chain, retriever = build_rag_chain(vectorstore)

    faithfulness_scores = []
    relevancy_scores = []
    recall_scores = []

    print(f"\n🧪 Evaluating {len(eval_dataset)} questions...\n")
    print("-" * 60)

    for i, item in enumerate(eval_dataset):
        question = item["question"]
        ground_truth = item["ground_truth"]

        print(f"[{i+1}/{len(eval_dataset)}] {question}")

        chunks = retriever.invoke(question)
        answer = chain.invoke(question)

        f = score_faithfulness(answer, chunks)
        r = score_relevancy(question, answer)
        c = score_context_recall(question, chunks, ground_truth)

        faithfulness_scores.append(f)
        relevancy_scores.append(r)
        recall_scores.append(c)

        print(f"  Answer:       {answer[:90]}...")
        print(f"  Faithfulness: {f} | Relevancy: {r} | Context Recall: {c}\n")

    print("=" * 60)
    print("         FINAL EVALUATION RESULTS")
    print("=" * 60)
    print(f"  Faithfulness:     {np.mean(faithfulness_scores):.3f}  (1.0 = perfect)")
    print(f"  Answer Relevancy: {np.mean(relevancy_scores):.3f}  (1.0 = perfect)")
    print(f"  Context Recall:   {np.mean(recall_scores):.3f}  (1.0 = perfect)")
    print("=" * 60)
    print("\n💡 Try changing chunk_size in ingest.py and re-run to see scores shift")
    
    # write results to file for CI artifact
    import json
    os.makedirs("logs", exist_ok=True)
    with open("logs/eval_results.json", "w") as f:
        json.dump({
            "faithfulness": float(np.mean(faithfulness_scores)),
            "answer_relevancy": float(np.mean(relevancy_scores)),
            "context_recall": float(np.mean(recall_scores))
        }, f, indent=2)
    print("\n✅ Results saved to logs/eval_results.json")
     
if __name__ == "__main__":
    run_evaluation()

