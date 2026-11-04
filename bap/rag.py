"""Retrieval-augmented Q&A over unstructured documents.

Retrieval uses TF-IDF (CPU-only, milliseconds, no embedding API calls) - a
deliberate green choice for a corpus of this size. Only the top-k chunks are
sent to the LLM, which keeps prompts (and energy) small.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from . import llm

RAG_SYSTEM = (
    "You answer questions for executives of Aurora Meridian Group using ONLY the provided "
    "document excerpts. Cite sources as [doc title]. If the excerpts do not contain the answer, "
    "say so. Be concise (max 6 sentences or a short bullet list)."
)


@dataclass
class Chunk:
    doc_id: str
    title: str
    text: str


def chunk_documents(docs: list[dict], max_words: int = 160) -> list[Chunk]:
    """One chunk per '## ' section; long sections are split on paragraph boundaries."""
    chunks: list[Chunk] = []
    for d in docs:
        body = re.sub(r"^# .*\n", "", d["text"]).strip()
        for sec in re.split(r"\n(?=## )", "\n" + body):
            sec = sec.strip()
            if not sec:
                continue
            lines = sec.split("\n", 1)
            heading = lines[0].lstrip("# ").strip() if lines[0].startswith("##") else ""
            text = lines[1] if heading and len(lines) > 1 else sec
            paras = [x.strip() for x in re.split(r"\n\s*\n", text)
                     if x.strip() and not (x.strip().startswith("*") and x.strip().endswith("*"))]
            buf: list[str] = []
            for para in paras:
                if buf and sum(len(b.split()) for b in buf) + len(para.split()) > max_words:
                    chunks.append(Chunk(d["id"], d["title"], (heading + ": " if heading else "") + "\n".join(buf)))
                    buf = []
                buf.append(para)
            if buf:
                chunks.append(Chunk(d["id"], d["title"], (heading + ": " if heading else "") + "\n".join(buf)))
    return chunks


class Retriever:
    def __init__(self, docs: list[dict]):
        self.chunks = chunk_documents(docs)
        self.vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
        self.matrix = self.vec.fit_transform([f"{c.title} {c.text}" for c in self.chunks])

    def search(self, query: str, k: int = 4) -> list[tuple[Chunk, float]]:
        sims = cosine_similarity(self.vec.transform([query]), self.matrix).ravel()
        idx = np.argsort(-sims)[:k]
        return [(self.chunks[i], float(sims[i])) for i in idx if sims[i] > 0.01]


def _extractive(query: str, hits: list[tuple[Chunk, float]]) -> str:
    if not hits:
        return "No relevant passage found in the document library."
    terms = {w for w in re.findall(r"[a-z]{4,}", query.lower())}
    best = []
    for c, _ in hits[:3]:
        clean = re.sub(r"^[^:]{0,60}:\s*", "", c.text)  # drop section heading prefix
        sents = [x for x in re.split(r"(?<=[.!?])\s+|\n", re.sub(r"[#*]", "", clean)) if len(x.split()) > 4]
        scored = sorted(sents, key=lambda s: -sum(t in s.lower() for t in terms))
        if scored and scored[0].strip():
            best.append(f"- {scored[0].strip()} [{c.title}]")
    return "Most relevant statements (offline extractive mode):\n" + "\n".join(best)


def answer(query: str, retriever: Retriever, use_llm: bool = True, k: int = 4) -> dict:
    hits = retriever.search(query, k=k)
    notes = []
    if use_llm and llm.is_configured() and hits:
        context = "\n\n".join(f"[{c.title}]\n{c.text}" for c, _ in hits)
        try:
            g = llm.generate(f"Excerpts:\n{context}\n\nQuestion: {query}", system=RAG_SYSTEM, max_tokens=450)
            return {"answer": g["text"], "hits": hits, "mode": "gemini (cached)" if g["cached"] else "gemini", "notes": notes}
        except llm.LLMUnavailable as e:
            notes.append(f"Gemini unavailable: {e}")
    from . import green
    green.record(offline_answers=1)
    return {"answer": _extractive(query, hits), "hits": hits, "mode": "offline", "notes": notes}
