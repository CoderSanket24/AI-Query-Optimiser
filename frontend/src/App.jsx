import { useState, useEffect, useCallback } from 'react'
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, BarChart, Bar, Legend
} from 'recharts'
import {
  executeQuery, fetchSummary, fetchHistory,
  fetchRewards, fetchPpoStats, fetchAnomalyStatus
} from './api/api'
import './App.css'

// ── Pill colours for join order ────────────────────────────────────────
const PILL_CLASSES = ['pill-1', 'pill-2', 'pill-3', 'pill-4']

// ─────────────────────────────────────────────────────────────────────
// Tab: Query Runner
// ─────────────────────────────────────────────────────────────────────
function QueryRunner() {
  const [sql, setSql]       = useState("SELECT t.title, n.name FROM title t JOIN cast_info ci ON t.id=ci.movie_id JOIN name n ON ci.person_id=n.id WHERE t.production_year=2000 LIMIT 10")
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError]   = useState(null)

  const run = async () => {
    setLoading(true); setError(null); setResult(null)
    try {
      const data = await executeQuery(sql)
      setResult(data)
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  const xai = result?.xai_explanation || {}
  const maxAttn = Math.max(...Object.values(xai), 1)

  return (
    <div>
      <div className="section">
        <h2>Run a Query</h2>
        <textarea
          className="query-input"
          value={sql}
          onChange={e => setSql(e.target.value)}
          placeholder="Enter a SELECT query with JOINs..."
        />
        <button className="run-btn" onClick={run} disabled={loading}>
          {loading && <span className="spinner" />}
          {loading ? 'Optimizing...' : 'Run Query'}
        </button>
        {error && <div className="error-msg">Error: {error}</div>}
      </div>

      {result && (
        <div className="section">
          <h2>AI Optimization Result</h2>
          <div className="result-box">
            <h3>Execution Metrics</h3>

            <div className="result-row">
              <span className="label">Exec Time</span>
              <span className="value">{result.exec_time_ms} ms</span>
            </div>
            <div className="result-row">
              <span className="label">Wait Time</span>
              <span className="value">{result.wait_time_ms} ms</span>
            </div>
            <div className="result-row">
              <span className="label">Total Latency</span>
              <span className="value">{result.latency_ms} ms</span>
            </div>
            <div className="result-row">
              <span className="label">Active Connections</span>
              <span className="value">{result.active_connections}</span>
            </div>
            <div className="result-row">
              <span className="label">Anomaly Detected</span>
              <span className={`anomaly-badge ${result.anomaly_detected ? 'anomaly-true' : 'anomaly-false'}`}>
                {result.anomaly_detected ? 'YES — DoS Pattern' : 'NO — Normal'}
              </span>
            </div>

            <h3 style={{ marginTop: 20, marginBottom: 12 }}>AI Join Order (XAI)</h3>
            <div className="result-row">
              <span className="label">Chosen Order</span>
              <div className="order-pills">
                {(result.choosen_order || []).map((t, i) => (
                  <span key={t} className={`pill ${PILL_CLASSES[i % 4]}`}>{i + 1}. {t}</span>
                ))}
              </div>
            </div>
            <div style={{ marginTop: 16 }}>
              {Object.entries(xai).sort(([, a], [, b]) => b - a).map(([table, pct]) => (
                <div className="attention-bar-row" key={table}>
                  <div className="attention-label">
                    <span>{table}</span>
                    <span>{pct}%</span>
                  </div>
                  <div className="attention-bar-bg">
                    <div className="attention-bar-fill" style={{ width: `${(pct / maxAttn) * 100}%` }} />
                  </div>
                </div>
              ))}
            </div>

            <h3 style={{ marginTop: 20, marginBottom: 8 }}>Optimized SQL</h3>
            <pre style={{ background: '#0f172a', padding: 12, borderRadius: 6, fontSize: '0.78rem', color: '#a5f3fc', overflowX: 'auto', whiteSpace: 'pre-wrap', wordBreak: 'break-all' }}>
              {result.optimized_query}
            </pre>
          </div>
        </div>
      )}
    </div>
  )
}

// ─────────────────────────────────────────────────────────────────────
// Tab: Dashboard (summary cards + reward chart)
// ─────────────────────────────────────────────────────────────────────
function Dashboard() {
  const [summary,  setSummary]  = useState(null)
  const [rewards,  setRewards]  = useState([])
  const [ppo,      setPpo]      = useState(null)
  const [loading,  setLoading]  = useState(true)

  const load = useCallback(async () => {
    setLoading(true)
    try {
      const [s, r, p] = await Promise.all([fetchSummary(), fetchRewards(), fetchPpoStats()])
      setSummary(s)
      // recharts expects array; rewards may be {rewards:[...]} or [...]
      setRewards(Array.isArray(r) ? r : (r.rewards || []))
      setPpo(p)
    } catch (e) {
      console.error(e)
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => { load() }, [load])

  const chartData = rewards.slice(-50).map((r, i) => ({
    index: i + 1,
    reward: typeof r === 'object' ? r.reward : r,
  }))

  return (
    <div>
      <button className="refresh-btn" onClick={load}>Refresh</button>

      {loading && <p style={{ color: '#64748b' }}><span className="spinner" />Loading dashboard...</p>}

      {ppo && (
        <div className="cards-grid">
          <div className="card teal">
            <div className="card-label">PPO Train Steps</div>
            <div className="card-value">{ppo.train_steps?.toLocaleString()}</div>
            <div className="card-sub">Neural network updates</div>
          </div>
          <div className="card green">
            <div className="card-label">Avg Reward (Baseline)</div>
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
            <div className="card-sub">Isolation Forest trained on</div>
          </div>
        </div>
      )}

      {summary && (
        <div className="cards-grid">
          <div className="card">
            <div className="card-label">Total Queries</div>
            <div className="card-value">{summary.total_queries ?? '-'}</div>
          </div>
          <div className="card green">
            <div className="card-label">Avg Exec Time</div>
            <div className="card-value">{summary.avg_exec_time_ms?.toFixed(1) ?? '-'} ms</div>
          </div>
          <div className="card red">
            <div className="card-label">Anomalies Caught</div>
            <div className="card-value">{summary.total_anomalies ?? '-'}</div>
          </div>
          <div className="card teal">
            <div className="card-label">Avg Latency</div>
            <div className="card-value">{summary.avg_latency_ms?.toFixed(1) ?? '-'} ms</div>
          </div>
        </div>
      )}

      {chartData.length > 0 && (
        <div className="section">
          <h2>PPO Reward Learning Curve (last 50 queries)</h2>
          <div className="chart-wrap">
            <ResponsiveContainer width="100%" height={260}>
              <LineChart data={chartData} margin={{ top: 5, right: 20, left: 0, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#dcfce7" />
                <XAxis dataKey="index" stroke="#6aaa7e" tick={{ fontSize: 11 }} label={{ value: 'Query #', position: 'insideBottom', offset: -2, fill: '#6aaa7e', fontSize: 11 }} />
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
// Tab: History Table
// ─────────────────────────────────────────────────────────────────────
function HistoryTab() {
  const [records, setRecords] = useState([])
  const [loading, setLoading] = useState(true)

  const load = useCallback(async () => {
    setLoading(true)
    try {
      const data = await fetchHistory()
      setRecords(Array.isArray(data) ? data : (data.records || []))
    } catch (e) { console.error(e) }
    finally { setLoading(false) }
  }, [])

  useEffect(() => { load() }, [load])

  return (
    <div className="section">
      <h2>Query History</h2>
      <button className="refresh-btn" onClick={load}>Refresh</button>
      {loading && <p style={{ color: '#64748b' }}><span className="spinner" />Loading...</p>}
      <div style={{ overflowX: 'auto' }}>
        <table className="history-table">
          <thead>
            <tr>
              <th>ID</th>
              <th>Tables</th>
              <th>Exec (ms)</th>
              <th>Wait (ms)</th>
              <th>Reward</th>
              <th>PPO Step</th>
              <th>Anomaly</th>
            </tr>
          </thead>
          <tbody>
            {records.slice().reverse().map((r, i) => (
              <tr key={i}>
                <td>{r.query_id ?? '-'}</td>
                <td style={{ maxWidth: 200, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                  {Array.isArray(r.tables) ? r.tables.join(', ') : '-'}
                </td>
                <td>{r.exec_time_ms ?? '-'}</td>
                <td>{r.wait_time_ms ?? '-'}</td>
                <td style={{ fontFamily: 'monospace', color: (r.reward ?? 0) < -500 ? '#f87171' : '#4ade80' }}>
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
              <tr><td colSpan={7} style={{ color: '#475569', textAlign: 'center', padding: 24 }}>No records yet — run some queries!</td></tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  )
}

// ─────────────────────────────────────────────────────────────────────
// Tab: Anomaly / Isolation Forest Status
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
      <button className="refresh-btn" onClick={load}>Refresh</button>
      {loading && <p style={{ color: '#64748b' }}><span className="spinner" />Loading...</p>}
      {status && (
        <>
          <div className="cards-grid" style={{ marginBottom: 24 }}>
            <div className={`card ${status.is_fitted ? 'green' : 'red'}`}>
              <div className="card-label">Forest Status</div>
              <div className="card-value">{status.is_fitted ? 'FITTED' : 'NOT FITTED'}</div>
              <div className="card-sub">Isolation Forest</div>
            </div>
            <div className="card blue">
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
              <div className="status-item"><div className="s-label">Feature Vector</div><div className="s-value">{JSON.stringify(status.feature_vector)}</div></div>
              <div className="status-item"><div className="s-label">exec_time Excluded</div><div className="s-value" style={{ color: '#4ade80' }}>{status.exec_time_excluded}</div></div>
              <div className="status-item"><div className="s-label">Detection Pattern</div><div className="s-value">{status.detection_pattern}</div></div>
              <div className="status-item"><div className="s-label">Anomaly Action</div><div className="s-value">{status.anomaly_action}</div></div>
              <div className="status-item"><div className="s-label">Min Samples to Fit</div><div className="s-value">{status.min_samples_to_fit}</div></div>
              <div className="status-item"><div className="s-label">Retrain Every N</div><div className="s-value">{status.retrain_every_n}</div></div>
              <div className="status-item"><div className="s-label">Placement</div><div className="s-value">{status.placement}</div></div>
              <div className="status-item"><div className="s-label">Normal Reward Formula</div><div className="s-value">{status.normal_reward}</div></div>
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
        <span className="badge">Live</span>
      </div>

      <div className="tabs">
        {TABS.map((t, i) => (
          <button
            key={t}
            className={`tab-btn ${activeTab === i ? 'active' : ''}`}
            onClick={() => setActiveTab(i)}
          >
            {t}
          </button>
        ))}
      </div>

      {activeTab === 0 && <QueryRunner />}
      {activeTab === 1 && <Dashboard />}
      {activeTab === 2 && <HistoryTab />}
      {activeTab === 3 && <AnomalyTab />}
    </div>
  )
}
