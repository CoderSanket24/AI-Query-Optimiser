package com.aioptimizer.middleware.service;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.stereotype.Service;

import javax.sql.DataSource;
import java.sql.Connection;
import java.sql.Statement;
import java.sql.ResultSet;

/**
 * QueryTelemetryService
 * ----------------------
 * Measures three separate timing metrics for every query execution:
 *
 *   wait_time_ms  -- time to acquire a JDBC connection from HikariCP pool.
 *                    High wait + low connections = DB under DoS stress.
 *
 *   exec_time_ms  -- time PostgreSQL spends actually running the query
 *                    (total_wall_time - wait_time).
 *                    This is what the PPO join order affects.
 *
 *   latency_ms    -- total wall-clock time (wait + exec), kept for compatibility.
 *
 * active_connections -- current number of active sessions in pg_stat_activity.
 *
 * Isolation Forest (FastAPI) uses [wait_time_ms, exec_time_ms, active_connections]
 * to detect DoS patterns (high wait or exec with low connections).
 *
 * PPO reward uses exec_time_ms only, since join order only affects execution,
 * not connection pool wait.
 */
@Service
public class QueryTelemetryService {

    @Autowired
    private JdbcTemplate jdbcTemplate;

    @Autowired
    private DataSource dataSource;

    public TelemetryResult executeAndTrack(String finalSql) throws Exception {

        // 1. Count active connections BEFORE execution (baseline load snapshot)
        String contentionQuery =
            "SELECT count(*) FROM pg_stat_activity WHERE state = 'active';";
        Integer activeConnections =
            jdbcTemplate.queryForObject(contentionQuery, Integer.class);

        // 2. Measure wait_time: time to acquire connection from HikariCP pool
        long waitStart = System.currentTimeMillis();
        Connection conn = dataSource.getConnection();
        long waitTimeMs = System.currentTimeMillis() - waitStart;

        // 3. Measure exec_time: time PostgreSQL spends running the query
        long execStart = System.currentTimeMillis();
        try (Statement stmt = conn.createStatement()) {
            // Use EXPLAIN ANALYZE to get real execution stats without returning rows
            stmt.execute("EXPLAIN ANALYZE " + finalSql);
        } finally {
            conn.close();   // return connection to pool
        }
        long execTimeMs = System.currentTimeMillis() - execStart;

        long latencyMs = waitTimeMs + execTimeMs;

        System.out.println("[Telemetry] wait=" + waitTimeMs + "ms"
            + "  exec=" + execTimeMs + "ms"
            + "  total=" + latencyMs + "ms"
            + "  connections=" + activeConnections);

        return new TelemetryResult(latencyMs, execTimeMs, waitTimeMs, activeConnections);
    }

    // ── Result holder ──────────────────────────────────────────────────────────
    public static class TelemetryResult {
        public long latencyMs;      // total  = wait + exec
        public long execTimeMs;     // PPO reward signal  (join order affects this)
        public long waitTimeMs;     // Isolation Forest signal (DoS detection)
        public int  activeConnections;

        public TelemetryResult(long latencyMs, long execTimeMs,
                               long waitTimeMs, int activeConnections) {
            this.latencyMs         = latencyMs;
            this.execTimeMs        = execTimeMs;
            this.waitTimeMs        = waitTimeMs;
            this.activeConnections = activeConnections;
        }
    }
}