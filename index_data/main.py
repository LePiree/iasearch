"""Indexation du dataset dans ElasticSearch (TD1-3).

Lit les lignes depuis pCloud, calcule leur embedding via Ollama, les regroupe
par KMeans, puis indexe :
  - un index par cluster (data_1, data_2, ...) : {"embedding": [...], "line": "..."}
  - un index "data_clusters" : {"embedding": <centroïde>, "data_index": "data_X"}

Ce script tourne sur la machine hôte : il utilise localhost par défaut.
"""

import os

import numpy as np
import requests
from elasticsearch import Elasticsearch, helpers
from sklearn.cluster import KMeans

ELASTICSEARCH_HOST = os.getenv("ELASTICSEARCH_HOST", "http://localhost:9200")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "qwen3-embedding:0.6b")
N_CLUSTERS = int(os.getenv("N_CLUSTERS", "10"))
BATCH_SIZE = 32

PCLOUD_API = (
    "https://apilux3.pcloud.com/getpublinkdownload"
    "?fileid=93128819642&code=XZnqkRJZ98VUd3rEuMXAFIsEpAtnNHoKYK8k"
    "&linkpassword=&forcedownload=1"
)
CLUSTERS_INDEX = "data_clusters"


def get_data_from_cloud() -> list[str]:
    """Récupère le lien de téléchargement (champs "path" et premier "hosts") puis le fichier."""
    meta = requests.get(PCLOUD_API, timeout=30).json()
    if meta.get("result") != 0:
        raise RuntimeError(f"Erreur pCloud : {meta}")
    url = f"https://{meta['hosts'][0]}{meta['path']}"
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    resp.encoding = "utf-8"
    lines = [
        " ".join(line.replace("<br />", " ").split())
        for line in resp.text.splitlines()
        if line.strip()
    ]
    print(f"{len(lines)} lignes téléchargées depuis {url}")
    return lines


def get_data_embeddings(lines: list[str]) -> np.ndarray:
    """Calcule l'embedding de chaque ligne via le docker Ollama (API /api/embed, par lots)."""
    embeddings: list[list[float]] = []
    for i in range(0, len(lines), BATCH_SIZE):
        batch = lines[i : i + BATCH_SIZE]
        resp = requests.post(
            f"{OLLAMA_HOST}/api/embed",
            json={"model": EMBEDDING_MODEL, "input": batch},
            timeout=300,
        )
        resp.raise_for_status()
        embeddings.extend(resp.json()["embeddings"])
        print(f"  embeddings : {len(embeddings)}/{len(lines)}", end="\r")
    print()
    return np.array(embeddings, dtype=np.float32)


def cluster_data(line_embeddings: np.ndarray, lines: list[str]):
    """KMeans sur les embeddings. Retourne {cluster_id: [indices des lignes]} et les centroïdes."""
    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
    labels = kmeans.fit_predict(line_embeddings)
    cluster_to_lines: dict[int, list[int]] = {}
    for idx, label in enumerate(labels):
        cluster_to_lines.setdefault(int(label), []).append(idx)
    for cid in sorted(cluster_to_lines):
        sample = " | ".join(lines[i][:60] for i in cluster_to_lines[cid][:3])
        print(f"  cluster {cid + 1:2d} ({len(cluster_to_lines[cid]):3d} lignes) : {sample}")
    return cluster_to_lines, kmeans.cluster_centers_


def create_index(es: Elasticsearch, name: str, properties: dict) -> None:
    """(Re)crée un index avec son mapping AVANT l'indexation."""
    if es.indices.exists(index=name):
        es.indices.delete(index=name)
    es.indices.create(
        index=name,
        settings={"number_of_replicas": 0},  # nœud unique : pas de réplique
        mappings={"properties": properties},
    )


def embedding_mapping(dims: int) -> dict:
    return {"type": "dense_vector", "dims": dims, "index": True, "similarity": "cosine"}


def main():
    es = Elasticsearch(ELASTICSEARCH_HOST)
    print(f"ElasticSearch {es.info()['version']['number']} sur {ELASTICSEARCH_HOST}")

    lines = get_data_from_cloud()
    line_embeddings = get_data_embeddings(lines)
    dims = line_embeddings.shape[1]
    print(f"Embeddings calculés : {line_embeddings.shape}")
    cluster_to_lines, centroids = cluster_data(line_embeddings, lines)

    # Supprime d'anciens index data_* devenus orphelins (ex. si N_CLUSTERS a diminué)
    for old in es.indices.get(index="data_*", ignore_unavailable=True):
        es.indices.delete(index=old)

    # 1 index par cluster : data_1, data_2, ...
    for cid, line_ids in sorted(cluster_to_lines.items()):
        index_name = f"data_{cid + 1}"
        create_index(
            es,
            index_name,
            {
                "embedding": embedding_mapping(dims),
                # analyseur français pour le BM25 (stemming, stop words)
                "line": {"type": "text", "analyzer": "french"},
            },
        )
        actions = (
            {
                "_index": index_name,
                "_source": {"embedding": line_embeddings[i].tolist(), "line": lines[i]},
            }
            for i in line_ids
        )
        helpers.bulk(es, actions, refresh=True)
        print(f"  {index_name} : {len(line_ids)} documents indexés")

    # Index des clusters : embedding du centroïde + nom de l'index correspondant
    create_index(
        es,
        CLUSTERS_INDEX,
        {
            "embedding": embedding_mapping(dims),
            "data_index": {"type": "keyword"},
            "sample_lines": {"type": "text", "index": False},
        },
    )
    actions = (
        {
            "_index": CLUSTERS_INDEX,
            "_id": f"data_{cid + 1}",
            "_source": {
                "embedding": (centroids[cid] / np.linalg.norm(centroids[cid])).tolist(),
                "data_index": f"data_{cid + 1}",
                "sample_lines": [lines[i] for i in line_ids[:3]],
            },
        }
        for cid, line_ids in sorted(cluster_to_lines.items())
    )
    helpers.bulk(es, actions, refresh=True)
    print(f"{CLUSTERS_INDEX} : {len(cluster_to_lines)} clusters indexés")


if __name__ == "__main__":
    main()
