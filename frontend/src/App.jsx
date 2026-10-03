import { useState, useEffect, useCallback } from 'react'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts'
import { executeQuery, fetchSummary, fetchHistory, fetchPpoStats, fetchAnomalyStatus } from './api/api'
import './App.css'

const PILL_CLASSES = ['pill-1', 'pill-2', 'pill-3', 'pill-4']

// ─────────────────────────────────────────────────────────────────────
// Tab: Query Runner  (two-column layout)
// ─────────────────────────────────────────────────────────────────────
function QueryRunner() {
  const [sql, setSql]         = useState("SELECT t.title, n.name FROM title t JOIN cast_info ci ON t.id=ci.movie_id JOIN name n ON ci.person_id=n.id WHERE t.production_year=2000 LIMIT 10")
  const [result, setResult]   = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError]     = useState(null)

  const run = async () => {
    setLoading(true); setError(null); setResult(null)
    try   { setResult(await executeQuery(sql)) }
    catch (e) { setError(e.message) }
    finally   { setLoading(false) }
  }

  const xai     = result?.xai_explanation || {}
  const maxAttn = Math.max(...Object.values(xai), 1)

  return (
    <div className="runner-grid">

      {/* ── LEFT: Query Input Panel ── */}
      <div className="section runner-left">
        <h2>Run a Query</h2>
        <textarea
          className="query-input"
          value={sql}
          onChange={e => setSql(e.target.value)}
          placeholder="Enter a SELECT query with JOINs..."
        />
        <button className="run-btn" onClick={run} disabled={loading}>
          {loading && <span className="spinner" />}
          {loading ? 'Optimizing...' : '▶  Run Query'}
        </button>
        {error && <div className="error-msg">Error: {error}</div>}

        <div className="sample-label">Quick Sample Queries</div>
        {[
          ["3-Table Join",  "SELECT t.title, n.name FROM title t JOIN cast_info ci ON t.id=ci.movie_id JOIN name n ON ci.person_id=n.id WHERE t.production_year=2000 LIMIT 10"],
          ["Keyword Join",  "SELECT t.title, k.keyword FROM title t JOIN movie_keyword mk ON t.id=mk.movie_id JOIN keyword k ON mk.keyword_id=k.id WHERE t.production_year=2005 LIMIT 10"],
          ["Company Join",  "SELECT t.title, cn.name FROM title t JOIN movie_companies mc ON t.id=mc.movie_id JOIN company_name cn ON mc.company_id=cn.id WHERE t.production_year=2010 LIMIT 10"],
        ].map(([label, q]) => (
          <button key={label} className="sample-btn" onClick={() => setSql(q)}>{label}</button>
        ))}
      </div>

      {/* ── RIGHT: Result Panel ── */}
      <div className="section runner-right">
        <h2>AI Optimization Result</h2>

        {!result && !loading && !error && (
          <div className="empty-result">
            <div className="empty-icon">⚡</div>
            <p>Run a query on the left to see the AI optimization result here.</p>
          </div>
        )}

        {loading && (
          <div className="empty-result">
            <span className="spinner" style={{ width: 32, height: 32, borderWidth: 3 }} />
            <p style={{ marginTop: 16, color: '#6aaa7e' }}>AI is optimizing your query...</p>
          </div>
        )}

        {error && <div className="error-msg">Error: {error}</div>}

        {result && (
          <div className="result-box">
            <h3>Execution Metrics</h3>
            <div className="result-row"><span className="label">Exec Time</span><span className="value">{result.exec_time_ms} ms</span></div>
            <div className="result-row"><span className="label">Wait Time</span><span className="value">{result.wait_time_ms} ms</span></div>
            <div className="result-row"><span className="label">Total Latency</span><span className="value">{result.latency_ms} ms</span></div>
            <div className="result-row"><span className="label">Connections</span><span className="value">{result.active_connections}</span></div>
            <div className="result-row">
              <span className="label">Anomaly</span>
              <span className={`anomaly-badge ${result.anomaly_detected ? 'anomaly-true' : 'anomaly-false'}`}>
                {result.anomaly_detected ? '🚨 DoS Detected' : '✅ Normal'}
              </span>
            </div>

            <h3 style={{ marginTop: 20, marginBottom: 10 }}>XAI — Attention Weights</h3>
            <div className="result-row" style={{ marginBottom: 12 }}>
              <span className="label">Join Order</span>
              <div className="order-pills">
                {(result.choosen_order || []).map((t, i) => (
                  <span key={t} className={`pill ${PILL_CLASSES[i % 4]}`}>{i + 1}. {t}</span>
                ))}
              </div>
            </div>
            {Object.entries(xai).sort(([, a], [, b]) => b - a).map(([table, pct]) => (
              <div className="attention-bar-row" key={table}>
                <div className="attention-label"><span>{table}</span><span>{pct}%</span></div>
                <div className="attention-bar-bg">
                  <div className="attention-bar-fill" style={{ width: `${(pct / maxAttn) * 100}%` }} />
                </div>
              </div>
            ))}

            <h3 style={{ marginTop: 20, marginBottom: 8 }}>Optimized SQL</h3>
            <pre className="sql-pre">{result.optimized_query}</pre>
          </div>
        )}
      </div>

    </div>
  )
}

// ─────────────────────────────────────────────────────────────────────
// Tab: Dashboard
// ─────────────────────────────────────────────────────────────────────
function Dashboard() {
  const [summary, setSummary] = useState(null)
  const [points,  setPoints]  = useState([])
  const [ppo,     setPpo]     = useState(null)
  const [loading, setLoading] = useState(true)
  const [error,   setError]   = useState(null)

  const load = useCallback(async () => {
    setLoading(true); setError(null)
    try {
      const [s, imp, p] = await Promise.all([
        fetchSummary(),
        fetch('/api/analytics/improvement?n=50').then(r => r.json()),
        fetchPpoStats(),
      ])
      setSummary(s)
      setPoints(imp.points || [])
      setPpo(p)
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => { load() }, [load])

  const chartData = points.map(r => ({ id: r.query_id, reward: r.reward }))

  return (
    <div>
      <button className="refresh-btn" onClick={load}>↻ Refresh</button>
      {loading && <p style={{ color: '#6aaa7e' }}><span className="spinner" />Loading...</p>}
      {error   && <p style={{ color: '#dc2626', marginBottom: 16 }}>Failed to load: {error}</p>}

      {ppo && (
        <div className="cards-grid">
          <div className="card teal">
            <div className="card-label">PPO Train Steps</div>
            <div className="card-value">{ppo.train_steps?.toLocaleString()}</div>
            <div className="card-sub">Neural network updates</div>
          </div>
          <div className="card green">
            <div className="card-label">Baseline Reward</div>
            <div className="card-value">{ppo.baseline?.toFixed(1)}</div>
            <div className="card-sub">Higher = faster queries</div>
          </div>
          <div className="card yellow">
            <div className="card-label">Buffer Size</div>
            <div className="card-value">{ppo.buffer_size}</div>
            <div className="card-sub">Replay experiences</div>
          </div>
          <div className="card">
            <div className="card-label">Forest Samples</div>
            <div className="card-value">{ppo.forest_samples}</div>
            <div className="card-sub">Isolation Forest</div>
          </div>
        </div>
      )}

      {summary && summary.total_queries > 0 && (
        <div className="cards-grid">
          <div className="card">
            <div className="card-label">Total Queries</div>
            <div className="card-value">{summary.total_queries}</div>
          </div>
          <div className="card green">
            <div className="card-label">Avg Latency</div>
            <div className="card-value">{summary.latency?.avg_ms ?? '-'} ms</div>
          </div>
          <div className="card red">
            <div className="card-label">Anomalies Caught</div>
            <div className="card-value">{summary.anomalous_queries ?? 0}</div>
          </div>
          <div className="card teal">
            <div className="card-label">Avg Reward</div>
            <div className="card-value">{summary.reward?.avg?.toFixed(1) ?? '-'}</div>
          </div>
        </div>
      )}

      {summary && summary.total_queries === 0 && !loading && (
        <div className="section" style={{ color: '#6aaa7e', textAlign: 'center', padding: 40 }}>
          No queries recorded yet — go to Query Runner and run some queries!
        </div>
      )}

      {chartData.length > 0 && (
        <div className="section">
          <h2>PPO Reward Learning Curve</h2>
          <div className="chart-wrap">
            <ResponsiveContainer width="100%" height={280}>
              <LineChart data={chartData} margin={{ top: 5, right: 20, left: 0, bottom: 20 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#dcfce7" />
                <XAxis dataKey="id" stroke="#6aaa7e" tick={{ fontSize: 11 }} label={{ value: 'Query ID', position: 'insideBottom', offset: -10, fill: '#6aaa7e', fontSize: 11 }} />
                <YAxis stroke="#6aaa7e" tick={{ fontSize: 11 }} />
                <Tooltip contentStyle={{ background: '#ffffff', border: '1px solid #bbf7d0', borderRadius: 8, boxShadow: '0 4px 12px rgba(0,0,0,0.08)' }} labelStyle={{ color: '#6aaa7e' }} itemStyle={{ color: '#16a34a' }} />
                <Line type="monotone" dataKey="reward" stroke="#16a34a" strokeWidth={2.5} dot={false} name="Reward" />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}
    </div>
  )
}

// ─────────────────────────────────────────────────────────────────────
// Tab: History
// ─────────────────────────────────────────────────────────────────────
function HistoryTab() {
  const [records, setRecords] = useState([])
  const [loading, setLoading] = useState(true)

  const load = useCallback(async () => {
    setLoading(true)
    try {
      const data = await fetchHistory()
      setRecords(data.records || [])
    } catch (e) { console.error(e) }
    finally { setLoading(false) }
  }, [])

  useEffect(() => { load() }, [load])

  return (
    <div className="section">
      <h2>Query History</h2>
      <button className="refresh-btn" onClick={load}>↻ Refresh</button>
      {loading && <p style={{ color: '#6aaa7e' }}><span className="spinner" />Loading...</p>}
      <div style={{ overflowX: 'auto' }}>
        <table className="history-table">
          <thead>
            <tr>
              <th>ID</th><th>Tables</th><th>Exec (ms)</th>
              <th>Wait (ms)</th><th>Reward</th><th>PPO Step</th><th>Anomaly</th>
            </tr>
          </thead>
          <tbody>
            {[...records].reverse().map((r, i) => (
              <tr key={i}>
                <td>{r.query_id ?? '-'}</td>
                <td style={{ maxWidth: 200, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                  {Array.isArray(r.tables) ? r.tables.join(', ') : '-'}
                </td>
                <td>{r.exec_time_ms ?? '-'}</td>
                <td>{r.wait_time_ms ?? '-'}</td>
                <td style={{ fontFamily: 'monospace', color: (r.reward ?? 0) < -500 ? '#dc2626' : '#16a34a', fontWeight: 600 }}>
                  {r.reward?.toFixed(2) ?? '-'}
                </td>
                <td>{r.ppo_step ?? '-'}</td>
                <td>
                  <span className={`anomaly-dot ${r.is_anomalous ? 'red' : 'green'}`} />
                  {' '}{r.is_anomalous ? 'Yes' : 'No'}
                </td>
              </tr>
            ))}
            {records.length === 0 && !loading && (
              <tr><td colSpan={7} style={{ color: '#6aaa7e', textAlign: 'center', padding: 32 }}>No records yet — run some queries!</td></tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  )
}

// ─────────────────────────────────────────────────────────────────────
// Tab: Anomaly / Isolation Forest
// ─────────────────────────────────────────────────────────────────────
function AnomalyTab() {
  const [status,  setStatus]  = useState(null)
  const [loading, setLoading] = useState(true)

  const load = useCallback(async () => {
    setLoading(true)
    try { setStatus(await fetchAnomalyStatus()) }
    catch (e) { console.error(e) }
    finally { setLoading(false) }
  }, [])

  useEffect(() => { load() }, [load])

  return (
    <div>
      <button className="refresh-btn" onClick={load}>↻ Refresh</button>
      {loading && <p style={{ color: '#6aaa7e' }}><span className="spinner" />Loading...</p>}
      {status && (
        <>
          <div className="cards-grid" style={{ marginBottom: 24 }}>
            <div className={`card ${status.is_fitted ? 'green' : 'red'}`}>
              <div className="card-label">Forest Status</div>
              <div className="card-value">{status.is_fitted ? 'FITTED' : 'NOT FITTED'}</div>
              <div className="card-sub">Isolation Forest</div>
            </div>
            <div className="card teal">
              <div className="card-label">Samples Trained On</div>
              <div className="card-value">{status.n_samples_trained}</div>
            </div>
            <div className="card yellow">
              <div className="card-label">Contamination Rate</div>
              <div className="card-value">{(status.contamination * 100).toFixed(0)}%</div>
              <div className="card-sub">Expected anomaly %</div>
            </div>
            <div className="card">
              <div className="card-label">Estimators</div>
              <div className="card-value">{status.n_estimators}</div>
              <div className="card-sub">Decision trees</div>
            </div>
          </div>

          <div className="section">
            <h2>Isolation Forest Configuration</h2>
            <div className="status-grid">
              {[
                ['Feature Vector',     JSON.stringify(status.feature_vector)],
                ['exec_time Excluded', status.exec_time_excluded],
                ['Detection Pattern',  status.detection_pattern],
                ['Anomaly Action',     status.anomaly_action],
                ['Min Samples to Fit', status.min_samples_to_fit],
                ['Retrain Every N',    status.retrain_every_n],
                ['Placement',          status.placement],
                ['Normal Reward',      status.normal_reward],
              ].map(([label, val]) => (
                <div className="status-item" key={label}>
                  <div className="s-label">{label}</div>
                  <div className="s-value">{val}</div>
                </div>
              ))}
            </div>
          </div>
        </>
      )}
    </div>
  )
}

// ─────────────────────────────────────────────────────────────────────
// Root App
// ─────────────────────────────────────────────────────────────────────
const TABS = ['Query Runner', 'Dashboard', 'History', 'Security (Forest)']

export default function App() {
  const [activeTab, setActiveTab] = useState(0)
  return (
    <div className="app">
      <div className="header">
        <div>
          <h1>AI Query Optimizer</h1>
          <p>PPO Reinforcement Learning + Isolation Forest DoS Protection</p>
        </div>
        <span className="badge">● Live</span>
      </div>
      <div className="tabs">
        {TABS.map((t, i) => (
          <button key={t} className={`tab-btn ${activeTab === i ? 'active' : ''}`} onClick={() => setActiveTab(i)}>{t}</button>
        ))}
      </div>
      {activeTab === 0 && <QueryRunner />}
      {activeTab === 1 && <Dashboard />}
      {activeTab === 2 && <HistoryTab />}
      {activeTab === 3 && <AnomalyTab />}
    </div>
  )
}
