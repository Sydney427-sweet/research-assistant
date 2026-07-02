import streamlit as st
from rag import ask, load_vectorstore, build_rag_chain

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

    # show user message
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # generate answer
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            from reranker import rerank
            raw_chucks = st.session_state.retriever.invoke(question)
            chunks = rerank(question, raw_chunks, top_k=4)
            answer = st.session_state.chain.invoke(question)

        st.markdown(answer)

        with st.expander("📄 View retrieved chunks"):
            for i, chunk in enumerate(chunks):
                st.markdown(f"**Chunk {i+1}:**")
                st.caption(chunk.page_content)
                st.divider()

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "chunks": chunks
    })
