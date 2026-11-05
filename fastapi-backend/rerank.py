"""Rerank des documents par un LLM via Ollama (TD3).

Pour chaque document, on demande au modèle « ce document est-il pertinent pour la
requête ? » avec une sortie JSON contrainte à {"answer": "yes"|"no"}. On lit les
logprobs du token de réponse, on additionne les probabilités (exp(logprob)) des
variantes de "yes" et de "no", et le score vaut p_yes / (p_yes + p_no).
Les appels sont lancés en parallèle (un appel par document).
"""

import asyncio
import math
import os

import httpx

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
RERANK_MODEL = os.getenv("RERANK_MODEL", "qwen3.5:0.8b")
RERANK_CONCURRENCY = int(os.getenv("RERANK_CONCURRENCY", "8"))
# ~8 tokens suffisent pour atteindre le token de réponse dans '{"answer": "yes"}'
RERANK_NUM_PREDICT = int(os.getenv("RERANK_NUM_PREDICT", "10"))
TOP_LOGPROBS = 20

ANSWER_FORMAT = {
    "type": "object",
    "properties": {"answer": {"type": "string", "enum": ["yes", "no"]}},
    "required": ["answer"],
}

SYSTEM_PROMPT = (
    "You are a relevance judge for a search engine of French public procurement "
    "notices. Given a search query and a document, decide if the document is "
    'relevant to the query. Answer only "yes" or "no".'
)


def _normalize(token: str) -> str:
    """'Yes', ' yes', '"yes', 'YES' -> 'yes'."""
    return token.strip().strip('"').strip().lower()


def _yes_no_probs(entry: dict) -> tuple[float, float]:
    """Somme des probabilités des variantes yes / no dans les top_logprobs d'une position."""
    p_yes = p_no = 0.0
    candidates = entry.get("top_logprobs") or [entry]
    for cand in candidates:
        tok = _normalize(cand.get("token", ""))
        if tok == "yes":
            p_yes += math.exp(cand["logprob"])
        elif tok == "no":
            p_no += math.exp(cand["logprob"])
    return p_yes, p_no


def score_from_logprobs(logprobs: list[dict], content: str = "") -> float:
    """Score p_yes / (p_yes + p_no) à partir des logprobs renvoyés par Ollama.

    1. On cherche la position du token de réponse (le token généré vaut yes/no).
    2. Sinon (Ollama ne renvoie parfois que la 1re position), on prend la première
       position dont les top_logprobs contiennent des variantes yes/no.
    3. En dernier recours, on se rabat sur le contenu de la réponse.
    """
    answer_entries = [e for e in logprobs if _normalize(e.get("token", "")) in ("yes", "no")]
    other_entries = [e for e in logprobs if e not in answer_entries]
    for entry in answer_entries + other_entries:
        p_yes, p_no = _yes_no_probs(entry)
        if p_yes + p_no > 0:
            return p_yes / (p_yes + p_no)
    return 1.0 if '"yes"' in content else 0.0


async def _score_document(
    client: httpx.AsyncClient, sem: asyncio.Semaphore, query: str, document: str
) -> float:
    payload = {
        "model": RERANK_MODEL,
        "stream": False,
        "think": False,
        "logprobs": True,
        "top_logprobs": TOP_LOGPROBS,
        "format": ANSWER_FORMAT,
        "options": {"temperature": 0, "num_predict": RERANK_NUM_PREDICT},
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Query: {query}\nDocument: {document}\n"
                "Is the document relevant to the query?",
            },
        ],
    }
    async with sem:
        resp = await client.post(f"{OLLAMA_HOST}/api/chat", json=payload)
    resp.raise_for_status()
    data = resp.json()
    return score_from_logprobs(data.get("logprobs") or [], data["message"]["content"])


async def rerank(
    query: str,
    documents: list[dict],
    client: httpx.AsyncClient,
    text_key: str = "line",
) -> list[dict]:
    """Ajoute "rerank_score" à chaque document et trie par score décroissant.

    Un appel au modèle par document, tous lancés en parallèle (asyncio.gather).
    """
    if not documents:
        return []
    sem = asyncio.Semaphore(RERANK_CONCURRENCY)
    scores = await asyncio.gather(
        *(_score_document(client, sem, query, doc[text_key]) for doc in documents)
    )
    ranked = [{**doc, "rerank_score": score} for doc, score in zip(documents, scores)]
    return sorted(ranked, key=lambda d: d["rerank_score"], reverse=True)
