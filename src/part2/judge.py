import json
import re
from collections.abc import Sequence
from typing import Any

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from part2.rag import format_docs

ANSWER_JUDGE_SYSTEM = (
    "You are a strict evaluator of question answering systems. "
    "Return ONLY a JSON object, no other text."
)

ANSWER_JUDGE_HUMAN = (
    "Question:\n{question}\n\n"
    "Context the system was given:\n{context}\n\n"
    "System answer:\n{answer}\n\n"
    "Score the answer:\n"
    "- \"faithfulness\": 2 = every claim is supported by the context, 1 = some claims are not "
    "supported, 0 = mostly unsupported or invented. Use null if the context is empty.\n"
    "- \"relevance\": 2 = directly answers the question (an explicit statement that the context has "
    "no answer counts as 2 if that is true), 1 = partially, 0 = does not answer.\n"
    "Respond as: {{\"faithfulness\": <0|1|2|null>, \"relevance\": <0|1|2>}}"
)

CONTEXT_JUDGE_HUMAN = (
    "Question:\n{question}\n\n"
    "Retrieved excerpts:\n{context}\n\n"
    "Which excerpts contain information that helps answer the question? "
    "Respond as: {{\"relevant\": [<excerpt numbers>]}}"
)

_ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [("system", ANSWER_JUDGE_SYSTEM), ("human", ANSWER_JUDGE_HUMAN)]
)
_CONTEXT_PROMPT = ChatPromptTemplate.from_messages(
    [("system", ANSWER_JUDGE_SYSTEM), ("human", CONTEXT_JUDGE_HUMAN)]
)
_JSON = re.compile(r"\{.*\}", flags=re.DOTALL)


def _parse_json(text: str) -> dict[str, Any] | None:
    match = _JSON.search(text)
    if match is None:
        return None
    try:
        out = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return out if isinstance(out, dict) else None


def judge_answer(
    llm: BaseChatModel, question: str, docs: Sequence[Document], answer: str
) -> dict[str, int | None]:
    context = format_docs(docs) if docs else "(empty - the system answered without retrieval)"
    raw = (_ANSWER_PROMPT | llm | StrOutputParser()).invoke(
        {"question": question, "context": context, "answer": answer}
    )
    parsed = _parse_json(raw) or {}
    faith = parsed.get("faithfulness")
    rel = parsed.get("relevance")
    return {
        "faithfulness": int(faith) if isinstance(faith, (int, float)) and docs else None,
        "relevance": int(rel) if isinstance(rel, (int, float)) else None,
    }


# precision@k = |trafne fragmenty| / |kontekst|
def judge_context(llm: BaseChatModel, question: str, docs: Sequence[Document]) -> float | None:
    if not docs:
        return None
    raw = (_CONTEXT_PROMPT | llm | StrOutputParser()).invoke(
        {"question": question, "context": format_docs(docs)}
    )
    parsed = _parse_json(raw)
    if parsed is None or not isinstance(parsed.get("relevant"), list):
        return None
    relevant = {int(i) for i in parsed["relevant"] if isinstance(i, (int, float)) and 1 <= i <= len(docs)}
    return len(relevant) / len(docs)
