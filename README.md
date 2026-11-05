# IA Search

Moteur de recherche sur ~1000 objets de consultations de marchés publics, inspiré
de la chaîne d'indexation de ChatGPT (`rerank400_cluster100_bm10_ann4096`) :

1. **Indexation** (`index_data/`) : embedding de chaque ligne avec Ollama
   (`qwen3-embedding:0.6b`), clustering KMeans (10 clusters), un index
   ElasticSearch par cluster (`data_1` … `data_10`) et un index `data_clusters`
   qui contient l'embedding du centroïde de chaque cluster.
2. **Recherche** (`fastapi-backend/`, endpoint `GET /search`) :
   - calcul de l'embedding de la requête via Ollama ;
   - recherche **knn** dans `data_clusters` pour trouver les clusters les plus proches ;
   - recherche **hybride** dans les index `data_X` trouvés : `bool` query avec
     `should: [match (BM25), knn]` ;
   - **rerank** LLM (`qwen3.5:0.8b`) : un appel par document, en parallèle, sortie
     JSON contrainte à `{"answer": "yes"|"no"}`, `think=false`, `logprobs=true`,
     `top_logprobs=20`. Les probabilités `exp(logprob)` des variantes de yes / no
     sont sommées, le score vaut `p_yes / (p_yes + p_no)`, et les résultats dont le
     score est inférieur à **0.4** sont supprimés.
3. **Front** (`react-front/`) : React + Vite + TypeScript, servi par nginx.

```
            ┌──────────────┐  /search  ┌──────────────────┐  knn + hybride ┌───────────────┐
navigateur ─┤ react-front  ├──────────►│ fastapi-backend  ├───────────────►│ elasticsearch │
            │  :3000       │           │  :8000 (+5678)   │                │  :9200        │
            └──────────────┘           └────────┬─────────┘                └───────▲───────┘
                                                │ embed + rerank                   │
                                         ┌──────▼──────┐   embeddings     ┌────────┴───────┐
                                         │   ollama    │◄─────────────────┤ index_data     │
                                         │   :11434    │                  │ (machine hôte) │
                                         └─────────────┘                  └────────────────┘
```

## Arborescence

| Chemin | Rôle |
|---|---|
| `docker-compose.yml` | Stack : elasticsearch, ollama, backend, front |
| `index_data/main.py` | Script d'indexation (tourne sur la machine hôte, utilise `localhost`) |
| `fastapi-backend/main.py` | API FastAPI : `/search`, `/clusters`, `/health` |
| `fastapi-backend/rerank.py` | Rerank LLM par logprobs |
| `fastapi-backend/Dockerfile` | Image `uv` Python 3.12 trixie, lancée avec debugpy |
| `react-front/` | Front React (Dockerfile multi-stage Node puis nginx) |
| `.vscode/launch.json` | Debug FastAPI en local et en « Remote Attach » sur le docker |

## Prérequis

- Docker Desktop (avec Docker Compose)
- [uv](https://docs.astral.sh/uv/) pour le script d'indexation
- Node.js 20+ (uniquement pour lancer le front hors Docker)
- **GPU NVIDIA** (facultatif) : le service `ollama` réserve le GPU via un bloc
  `deploy`. **Sans GPU NVIDIA, supprimez le bloc `deploy:` du service `ollama`**
  dans `docker-compose.yml` (Ollama tournera alors sur le CPU, plus lentement).
- Environ 4 Go de RAM libres pour Docker (ElasticSearch est limité à 1 Go de heap).

## Relancer tout de zéro

Toutes les commandes se lancent depuis la racine du projet `iasearch/`.

```bash
# 0. (optionnel) repartir d'une base vide : supprime conteneurs ET volumes (index + modèles)
docker compose down -v

# 1. Démarrer ElasticSearch et Ollama
docker compose up -d elasticsearch ollama
curl http://localhost:9200/          # -> "You Know, for Search"
curl http://localhost:11434/         # -> "Ollama is running"

# 2. Télécharger les modèles dans le docker Ollama (stockés dans le volume ollama_data)
docker exec ollama ollama pull qwen3-embedding:0.6b
docker exec ollama ollama pull qwen3.5:0.8b

# 3. Indexer les données (sur la machine hôte, via localhost:9200 et localhost:11434)
cd index_data
uv sync
uv run main.py
cd ..
curl "http://localhost:9200/_cat/indices/data*?v"   # data_1..data_10 + data_clusters

# 4. Construire et démarrer le backend et le front
docker compose up -d --build backend front
```

Ensuite :

- Front : <http://localhost:3000>
- Swagger du backend : <http://localhost:8000/docs>
- Exemple : <http://localhost:8000/search?query=nettoyage%20des%20locaux>

Une fois les volumes créés (index + modèles), un simple `docker compose up -d`
relance toute la stack. Il ne faut ré-indexer que si le volume `es_data` a été supprimé.

> Sous Windows, si la console affiche mal les accents pendant l'indexation, lancez
> `PYTHONIOENCODING=utf-8 uv run main.py` (Git Bash) ou
> `$env:PYTHONIOENCODING="utf-8"; uv run main.py` (PowerShell).

## API

`GET /search`

| Paramètre | Défaut | Description |
|---|---|---|
| `query` | (obligatoire) | Texte recherché |
| `clusters` | 2 | Nombre de clusters (index `data_X`) fouillés |
| `size` | 10 | Nombre maximum de résultats |
| `rerank` | `true` | Active le rerank LLM et le filtre `score < 0.4` |

Exemple de réponse :

```json
{
  "query": "nettoyage des locaux",
  "clusters": [{"data_index": "data_2", "score": 0.909, "sample_lines": ["..."]}],
  "results": [
    {"line": "Nettoyage des locaux de la Communauté de communes du Genevois",
     "data_index": "data_2", "es_score": 8.2, "rerank_score": 0.999}
  ],
  "removed": 0,
  "rerank": true
}
```

Autres endpoints : `GET /clusters?query=...&k=3` (knn seul sur `data_clusters`),
`GET /health` (état d'ElasticSearch et d'Ollama).

## Configuration (variables d'environnement)

| Variable | Défaut (local) | Valeur dans Docker |
|---|---|---|
| `ELASTICSEARCH_HOST` | `http://localhost:9200` | `http://elasticsearch:9200` |
| `OLLAMA_HOST` | `http://localhost:11434` | `http://ollama:11434` |
| `EMBEDDING_MODEL` | `qwen3-embedding:0.6b` | idem |
| `RERANK_MODEL` | `qwen3.5:0.8b` | idem |
| `RERANK_THRESHOLD` | `0.4` | idem |
| `RERANK_CANDIDATES` | `20` | documents envoyés au rerank avant filtrage |
| `RERANK_NUM_PREDICT` | `10` | tokens générés par appel de rerank |
| `N_CLUSTERS` (index_data) | `10` | nombre de clusters KMeans |

Dans Docker, le backend joint les autres services par leur nom de service. Le
script `index_data` tourne sur l'hôte et utilise `localhost`. Le front est
construit avec `VITE_API_URL=http://localhost:8000`, car c'est le navigateur qui
appelle le backend.

## Développement et debug

**Backend en local** (avec ElasticSearch et Ollama démarrés dans Docker) :

```bash
cd fastapi-backend
uv sync
uv run uvicorn main:app --reload --port 8000
```

Ou avec VSCode : configuration **« FastAPI (local) »**.

**Backend dans Docker** : le conteneur démarre uvicorn sous `debugpy` (port 5678).
Lancez la configuration **« FastAPI (Remote Attach docker) »** puis posez des points
d'arrêt dans `fastapi-backend/`. Le code est copié dans l'image (pas de volume
monté sur `/app`) : après une modification, reconstruisez l'image avec
`docker compose up -d --build backend`.

**Front en local** :

```bash
cd react-front
npm install
npm run dev     # http://localhost:5173
```

## Notes techniques

- **Mappings créés avant l'indexation** : `embedding` en `dense_vector`
  (1024 dimensions, `similarity: cosine`), `line` en `text` avec l'analyseur
  `french` (BM25 avec stemming), `data_index` en `keyword`. Les index n'ont aucune
  réplique (`number_of_replicas: 0`), ce qui les garde en `green` sur un nœud unique.
- **Logprobs avec Ollama** : avec `qwen3.5:0.8b`, Ollama (testé en 0.33.2) ne
  renvoie les logprobs que pour la première position générée. Le rerank cherche
  d'abord la position dont le token est yes/no. Sinon, il utilise la première
  position dont les `top_logprobs` contiennent des variantes de yes/no (`Yes`,
  `yes`, `No`, `no`…), qui reflètent directement le jugement du modèle. Le texte
  JSON généré n'est pas utilisé pour le score.
- **Parallélisme** : les appels de rerank partent tous en même temps
  (`asyncio.gather`, au plus 8 simultanés). Avec ce modèle, Ollama les sert l'un
  après l'autre (~0,3 s par document sur une RTX 4050), soit ~5 à 6 s par
  recherche avec rerank.
- Le modèle de 0,8B paramètres reste approximatif. Il élimine bien les documents
  hors sujet (score < 0,4), mais l'ordre entre documents proches du sujet est
  parfois discutable.
