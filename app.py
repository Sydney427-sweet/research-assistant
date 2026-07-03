import time
import streamlit as st
from rag import load_vectorstore, build_rag_chain
from reranker import rerank
from logger import log_query

st.set_page_config(
    page_title="Research Assistant",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Personal Research Assistant")
st.caption("Ask questions about your uploaded documents — powered by local AI")

# initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

if "chain" not in st.session_state:
    with st.spinner("Loading your documents..."):
        vectorstore = load_vectorstore()
        chain, retriever = build_rag_chain(vectorstore)
        st.session_state.chain = chain
        st.session_state.retriever = retriever

# display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and "chunks" in msg:
            with st.expander("📄 View retrieved chunks"):
                for i, chunk in enumerate(msg["chunks"]):
                    st.markdown(f"**Chunk {i+1}:**")
                    st.caption(chunk.page_content)
                    st.divider()

# chat input
if question := st.chat_input("Ask a question about your documents..."):

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            total_start = time.time()

            # retrieval
            t0 = time.time()
            raw_chunks = st.session_state.retriever.invoke(question)
            retrieval_ms = (time.time() - t0) * 1000

            # rerank
            t1 = time.time()
            chunks, scores = rerank(question, raw_chunks, top_k=4)
            rerank_ms = (time.time() - t1) * 1000

            # generation
            t2 = time.time()
            answer = st.session_state.chain.invoke(question)
            generation_ms = (time.time() - t2) * 1000

            total_ms = (time.time() - total_start) * 1000

            # log it
            log_query(
                question=question,
                answer=answer,
                chunks=chunks,
                latency={
                    "retrieval_ms": retrieval_ms,
                    "rerank_ms": rerank_ms,
                    "generation_ms": generation_ms,
                    "total_ms": total_ms,
                    "reranker_scores": scores
                }
            )

        st.markdown(answer)

        # show latency inline
        st.caption(
            f"⏱ Total: {total_ms:.0f}ms | "
            f"Retrieval: {retrieval_ms:.0f}ms | "
            f"Rerank: {rerank_ms:.0f}ms | "
            f"Generation: {generation_ms:.0f}ms"
        )

        with st.expander("📄 View retrieved chunks"):
            for i, (chunk, score) in enumerate(zip(chunks, scores)):
                st.markdown(f"**Chunk {i+1}** — rerank score: `{score}`")
                st.caption(chunk.page_content)
                st.divider()

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "chunks": chunks
    })
