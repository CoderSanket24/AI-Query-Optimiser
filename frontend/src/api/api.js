// api.js
// ------
// All calls go to Spring Boot :8080 ONLY.
// FastAPI (:8000) is never called from the browser.
// Spring Boot AnalyticsController proxies the analytics endpoints.

const BASE = '/api';  // Vite proxy forwards to http://localhost:8080/api

export const executeQuery = async (sql) => {
  const res = await fetch(`${BASE}/query/execute`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query: sql }),
  });
  if (!res.ok) {
    const err = await res.text();
    throw new Error(err);
  }
  return res.json();
};

export const fetchSummary = async () => {
  const res = await fetch(`${BASE}/analytics/summary`);
  if (!res.ok) throw new Error('Failed to fetch summary');
  return res.json();
};

export const fetchHistory = async () => {
  const res = await fetch(`${BASE}/analytics/history`);
  if (!res.ok) throw new Error('Failed to fetch history');
  return res.json();
};

export const fetchImprovement = async (n = 50) => {
  const res = await fetch(`${BASE}/analytics/improvement?n=${n}`);
  if (!res.ok) throw new Error('Failed to fetch improvement data');
  return res.json();
};

export const fetchPpoStats = async () => {
  const res = await fetch(`${BASE}/analytics/ppo/stats`);
  if (!res.ok) throw new Error('Failed to fetch PPO stats');
  return res.json();
};

export const fetchAnomalyStatus = async () => {
  const res = await fetch(`${BASE}/analytics/anomaly`);
  if (!res.ok) throw new Error('Failed to fetch anomaly status');
  return res.json();
};
