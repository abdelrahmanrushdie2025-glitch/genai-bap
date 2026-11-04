"""Retrieval-augmented Q&A over unstructured enterprise documents."""
import streamlit as st

from bap import data, rag, ui

ui.page("Document Insights", "📄")
st.title("📄 Document Insights")
st.caption("Ask about earnings reports, board memos, market analyses, audit findings and regulation. "
           "Answers are grounded in the retrieved passages and cite their sources.")

examples = [
    "What did internal audit find about vendor payments?",
    "What are our green computing principles?",
    "What does DORA require from us?",
    "Why did the operating margin narrow in Q2 2026?",
    "What savings does the efficiency programme target?",
]
pick = st.selectbox("Example questions", ["- choose -"] + examples)
q = st.text_input("Your question", value="" if pick == "- choose -" else pick)

if q:
    with st.spinner("Retrieving passages and drafting answer ..."):
        res = rag.answer(q, ui.retriever(), use_llm=ui.use_llm())
    st.markdown("### Answer")
    st.markdown(res["answer"])
    for n in res["notes"]:
        st.info(n)
    ui.mode_badge(res["mode"])
    st.markdown("### Sources")
    for chunk, score in res["hits"]:
        with st.expander(f"{chunk.title} - relevance {score:.2f}"):
            st.markdown(chunk.text)

st.divider()
st.subheader("Document library")
for d in data.load_documents():
    with st.expander(d["title"]):
        st.markdown(d["text"])
