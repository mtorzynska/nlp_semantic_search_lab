# RAG: prompt + LCEL chain (retriever -> context -> LLM)
# Ref: https://python.langchain.com/docs/tutorials/rag/

import time
from collections.abc import Iterable, Mapping
from typing import Any

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import Runnable, RunnableLambda, RunnableParallel, RunnablePassthrough

RAG_SYSTEM_PROMPT = (
    "You are an assistant that answers questions about movies using ONLY "
    "the movie review excerpts provided in the context.\n\n"
    "Rules:\n"
    "1. Base the answer strictly on the context. Do not use outside knowledge about films, actors or plots.\n"
    "2. Cite the excerpts you used with their numbers in square brackets, e.g. [1], [3].\n"
    "3. If the context does not contain the information needed, say clearly that the reviews "
    "do not answer the question - do not guess.\n"
    "4. Reviews are subjective opinions: report them as opinions (\"one reviewer thinks...\") "
    "and mention when reviewers disagree.\n"
    "5. Be concise (at most 5 sentences). Answer in the same language as the question."
)

RAG_HUMAN_PROMPT = (
    "Context (movie review excerpts):\n"
    "{context}\n\n"
    "Question: {question}\n\n"
    "Answer:"
)

# baseline bez kontekstu
NO_RAG_SYSTEM_PROMPT = (
    "You are an assistant that answers questions about movies. "
    "Be concise (at most 5 sentences). If you do not know the answer, say so. "
    "Answer in the same language as the question."
)

# LLM + RAG
HYBRID_SYSTEM_PROMPT = (
    "You are an assistant that answers questions about movies. You are given movie review "
    "excerpts as context.\n\n"
    "Rules:\n"
    "1. Use the excerpts first and cite them with their numbers in square brackets, e.g. [1], [3].\n"
    "2. If the excerpts are not enough, you may add your own general knowledge, but mark every "
    "such sentence with \"(general knowledge)\".\n"
    "3. Reviews are subjective opinions: report them as opinions (\"one reviewer thinks...\") "
    "and mention when reviewers disagree.\n"
    "4. Be concise (at most 5 sentences). Answer in the same language as the question."
)

RAG_PROMPT = ChatPromptTemplate.from_messages(
    [("system", RAG_SYSTEM_PROMPT), ("human", RAG_HUMAN_PROMPT)]
)
HYBRID_PROMPT = ChatPromptTemplate.from_messages(
    [("system", HYBRID_SYSTEM_PROMPT), ("human", RAG_HUMAN_PROMPT)]
)
NO_RAG_PROMPT = ChatPromptTemplate.from_messages(
    [("system", NO_RAG_SYSTEM_PROMPT), ("human", "{question}")]
)


# numerowane fragmenty [n] -> do cytowania w odpowiedzi
def format_docs(docs: Iterable[Document]) -> str:
    blocks = []
    for i, doc in enumerate(docs, start=1):
        meta = doc.metadata
        header = f"[{i}] (review #{meta.get('review_id')}, sentiment: {meta.get('label')})"
        blocks.append(f"{header}\n{doc.page_content}")
    return "\n\n".join(blocks) if blocks else "(no documents retrieved)"


# chains: question -> {"question", "docs", "answer"}
# prompt: RAG_PROMPT (tylko kontekst) albo HYBRID_PROMPT (LLM + RAG)
def build_rag_chain(
    retriever: BaseRetriever, llm: BaseChatModel, prompt: ChatPromptTemplate = RAG_PROMPT
) -> Runnable:
    generate = (
        RunnableLambda(lambda x: {"context": format_docs(x["docs"]), "question": x["question"]})
        | prompt
        | llm
        | StrOutputParser()
    )
    return RunnableParallel(docs=retriever, question=RunnablePassthrough()).assign(answer=generate)


def build_no_rag_chain(llm: BaseChatModel) -> Runnable:
    generate = NO_RAG_PROMPT | llm | StrOutputParser()
    return RunnableParallel(
        question=RunnablePassthrough(),
        docs=RunnableLambda(lambda _: []),
        answer=RunnableLambda(lambda q: {"question": q}) | generate,
    )


def run_variants(
    queries: Iterable[str], chains: Mapping[str, Runnable]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for query in queries:
        for variant, chain in chains.items():
            t0 = time.perf_counter()
            try:
                out = chain.invoke(query)
                answer, docs, error = out["answer"], out["docs"], None
            except Exception as exc:  # błąd API nie przerywa całego eksperymentu
                answer, docs, error = "", [], f"{type(exc).__name__}: {exc}"
            rows.append(
                {
                    "query": query,
                    "variant": variant,
                    "answer": answer,
                    "sources": [d.metadata.get("review_id") for d in docs],
                    "source_labels": [d.metadata.get("label") for d in docs],
                    "seconds": round(time.perf_counter() - t0, 2),
                    "error": error,
                    "docs": docs,
                }
            )
    return rows
