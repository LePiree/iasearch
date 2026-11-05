import { useState, type FormEvent } from 'react'
import './App.css'

const API_URL = import.meta.env.VITE_API_URL ?? 'http://localhost:8000'

type Cluster = { data_index: string; score: number; sample_lines: string[] }
type Result = {
  line: string
  data_index: string
  es_score: number
  rerank_score?: number
}
type SearchResponse = {
  query: string
  clusters: Cluster[]
  results: Result[]
  removed?: number
  rerank?: boolean
}

function App() {
  const [query, setQuery] = useState('')
  const [rerank, setRerank] = useState(true)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [data, setData] = useState<SearchResponse | null>(null)

  async function handleSearch(e: FormEvent) {
    e.preventDefault()
    if (!query.trim()) return
    setLoading(true)
    setError(null)
    try {
      const params = new URLSearchParams({ query, rerank: String(rerank) })
      const resp = await fetch(`${API_URL}/search?${params}`)
      if (!resp.ok) throw new Error(`HTTP ${resp.status} : ${await resp.text()}`)
      setData(await resp.json())
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
      setData(null)
    } finally {
      setLoading(false)
    }
  }

  return (
    <main className="app">
      <h1>IA Search</h1>
      <p className="subtitle">
        Recherche dans les objets de consultations publiques : clusters par
        embedding, recherche hybride BM25 + knn, rerank par LLM.
      </p>

      <form className="search" onSubmit={handleSearch}>
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="ex : nettoyage des locaux, rénovation d'une école…"
          autoFocus
        />
        <button type="submit" disabled={loading || !query.trim()}>
          {loading ? 'Recherche…' : 'Search'}
        </button>
      </form>
      <label className="toggle">
        <input
          type="checkbox"
          checked={rerank}
          onChange={(e) => setRerank(e.target.checked)}
        />
        Rerank LLM (supprime les résultats avec un score &lt; 0.4)
      </label>

      {error && <p className="error">{error}</p>}

      {data && (
        <>
          <section className="clusters">
            <h2>Clusters les plus proches</h2>
            <ul>
              {data.clusters.map((c) => (
                <li key={c.data_index} title={c.sample_lines.join('\n')}>
                  <strong>{c.data_index}</strong> · {c.score.toFixed(3)}
                </li>
              ))}
            </ul>
          </section>

          <section>
            <h2>
              {data.results.length} résultat(s)
              {data.removed ? (
                <span className="removed">
                  {' '}
                  · {data.removed} supprimé(s) par le rerank
                </span>
              ) : null}
            </h2>
            {data.results.length === 0 && (
              <p className="empty">Aucun document pertinent trouvé.</p>
            )}
            <ol className="results">
              {data.results.map((r, i) => (
                <li key={`${r.data_index}-${i}`}>
                  <p className="line">{r.line}</p>
                  <p className="meta">
                    <span>{r.data_index}</span>
                    <span>ES : {r.es_score.toFixed(2)}</span>
                    {r.rerank_score !== undefined && (
                      <span className="rerank">
                        rerank : {r.rerank_score.toFixed(3)}
                      </span>
                    )}
                  </p>
                </li>
              ))}
            </ol>
          </section>
        </>
      )}
    </main>
  )
}

export default App
