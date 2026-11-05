"""Backend de recherche : clusters par knn, puis recherche hybride dans les index data_X."""

import os
from contextlib import asynccontextmanager

import httpx
from elasticsearch import AsyncElasticsearch
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from rerank import rerank as rerank_documents

# En local : localhost. Dans docker compose, les variables d'environnement
# pointent vers http://elasticsearch:9200 et http://ollama:11434.
ELASTICSEARCH_HOST = os.getenv("ELASTICSEARCH_HOST", "http://localhost:9200")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "qwen3-embedding:0.6b")
CLUSTERS_INDEX = "data_clusters"
# En dessous de ce score de rerank, la requête et le document n'ont aucun rapport
RERANK_THRESHOLD = float(os.getenv("RERANK_THRESHOLD", "0.4"))
# Nombre de candidats envoyés au rerank (avant filtrage et troncature à "size")
RERANK_CANDIDATES = int(os.getenv("RERANK_CANDIDATES", "20"))
CORS_ORIGINS = os.getenv(
    "CORS_ORIGINS", "http://localhost:3000,http://localhost:5173"
).split(",")


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.es = AsyncElasticsearch(ELASTICSEARCH_HOST)
    app.state.http = httpx.AsyncClient(timeout=120)
    yield
    await app.state.es.close()
    await app.state.http.aclose()


app = FastAPI(title="IA Search", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:\d+)?",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def embed(text: str) -> list[float]:
    """Calcule l'embedding de la requête avec le même modèle que l'indexation."""
    try:
        resp = await app.state.http.post(
            f"{OLLAMA_HOST}/api/embed",
            json={"model": EMBEDDING_MODEL, "input": text},
        )
        resp.raise_for_status()
    except httpx.HTTPError as exc:
        raise HTTPException(502, f"Ollama embedding indisponible : {exc}") from exc
    return resp.json()["embeddings"][0]


async def search_clusters(query_vector: list[float], k: int) -> list[dict]:
    """Recherche vectorielle (knn) dans data_clusters."""
    resp = await app.state.es.search(
        index=CLUSTERS_INDEX,
        query={
            "knn": {
                "field": "embedding",
                "query_vector": query_vector,
                "k": k,
                "num_candidates": 50,
            }
        },
        size=k,
        source_excludes=["embedding"],
    )
    return [
        {
            "data_index": hit["_source"]["data_index"],
            "score": hit["_score"],
            "sample_lines": hit["_source"].get("sample_lines", []),
        }
        for hit in resp["hits"]["hits"]
    ]


async def search_documents(
    indices: list[str], query: str, query_vector: list[float], size: int
) -> list[dict]:
    """Recherche hybride : bool query combinant BM25 (match) et knn."""
    resp = await app.state.es.search(
        index=",".join(indices),
        query={
            "bool": {
                "should": [
                    {"match": {"line": {"query": query}}},
                    {
                        "knn": {
                            "field": "embedding",
                            "query_vector": query_vector,
                            "k": size,
                            "num_candidates": 100,
                        }
                    },
                ]
            }
        },
        size=size,
        source_excludes=["embedding"],
    )
    return [
        {
            "line": hit["_source"]["line"],
            "data_index": hit["_index"],
            "es_score": hit["_score"],
        }
        for hit in resp["hits"]["hits"]
    ]


@app.get("/health")
async def health():
    es_ok = await app.state.es.ping()
    try:
        ollama_ok = (await app.state.http.get(OLLAMA_HOST)).status_code == 200
    except httpx.HTTPError:
        ollama_ok = False
    return {"elasticsearch": es_ok, "ollama": ollama_ok}


@app.get("/clusters")
async def clusters(
    query: str = Query(..., min_length=1), k: int = Query(3, ge=1, le=20)
):
    """Clusters les plus proches de la requête (knn sur data_clusters)."""
    return await search_clusters(await embed(query), k)


@app.get("/search")
async def search(
    query: str = Query(..., min_length=1, description="Texte recherché"),
    clusters: int = Query(2, ge=1, le=10, description="Nombre de clusters à fouiller"),
    size: int = Query(10, ge=1, le=50, description="Nombre de documents retournés"),
    rerank: bool = Query(True, description="Rerank LLM + filtre score < seuil"),
):
    """1. knn sur data_clusters
    2. recherche hybride (match + knn) dans les index data_X retournés
    3. rerank LLM en parallèle, suppression des résultats sous RERANK_THRESHOLD
    """
    query_vector = await embed(query)
    found_clusters = await search_clusters(query_vector, clusters)
    response = {"query": query, "clusters": found_clusters, "results": [], "rerank": rerank}
    if not found_clusters:
        return response
    indices = [c["data_index"] for c in found_clusters]
    candidates = max(size, RERANK_CANDIDATES) if rerank else size
    results = await search_documents(indices, query, query_vector, candidates)

    if rerank:
        try:
            ranked = await rerank_documents(query, results, app.state.http)
        except httpx.HTTPError as exc:
            raise HTTPException(502, f"Ollama rerank indisponible : {exc}") from exc
        kept = [r for r in ranked if r["rerank_score"] >= RERANK_THRESHOLD]
        response["removed"] = len(ranked) - len(kept)
        results = kept

    response["results"] = results[:size]
    return response
