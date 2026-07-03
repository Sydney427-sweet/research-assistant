from sentence_transformers import CrossEncoder

model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def rerank(question: str, chunks: list, top_k: int = 4):
    pairs = [(question, chunk.page_content) for chunk in chunks]
    scores = model.predict(pairs)

    scored_chunks = sorted(
        zip(scores, chunks),
        key=lambda x: x[0],
        reverse=True
    )

    top_chunks = [chunk for _, chunk in scored_chunks[:top_k]]
    top_scores = [round(float(s), 3) for s, _ in scored_chunks[:top_k]]

    print(f"  🔀 Reranked {len(chunks)} chunks → kept top {top_k}")
    for i, (score, chunk) in enumerate(scored_chunks[:top_k]):
        print(f"     [{i+1}] score={score:.3f} | {chunk.page_content[:80]}...")

    return top_chunks, top_scores
