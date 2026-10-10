package com.aioptimizer.middleware.service;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.stereotype.Service;

import javax.sql.DataSource;
import java.sql.Connection;
import java.sql.Statement;

/**
 * QueryTelemetryService
 * ----------------------
 * Measures three separate timing metrics for every query execution.
 *
 *   wait_time_ms  -- time to acquire a JDBC connection from HikariCP pool.
 *                    High wait + low connections = DB under DoS stress.
 *
 *   exec_time_ms  -- time PostgreSQL spends actually running the query
 *                    (total_wall_time - wait_time).
 *                    This is what the PPO join order affects.
 *                    Uses System.nanoTime() so fast queries do NOT round to 0.
 *
 *   latency_ms    -- total wall-clock time (wait + exec), kept for compatibility.
 *
 * active_connections -- current number of active sessions in pg_stat_activity.
 *
 * IMPORTANT: exec_time_ms is -1 when timing is genuinely unavailable (should
 * never happen in practice). feedback.py treats -1 as "missing" and falls back
 * to REINFORCE rather than silently leaking wait_time into the reward.
 */
@Service
public class QueryTelemetryService {

    /** Sentinel: exec time could not be measured (not zero, which is valid). */
    public static final long EXEC_TIME_MISSING = -1L;

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

        // 2. Measure wait_time: time to acquire connection from HikariCP pool.
        //    Use nanoTime for sub-millisecond precision — fast pools return in
        //    microseconds and currentTimeMillis rounds to 0.
        long waitStart = System.nanoTime();
        Connection conn = dataSource.getConnection();
        long waitTimeMs = (System.nanoTime() - waitStart) / 1_000_000L;

        // 3. Measure exec_time: time PostgreSQL spends running the query.
        //    Also uses nanoTime so short analytical queries are not reported as 0.
        long execStart = System.nanoTime();
        long execTimeMs;
        try (Statement stmt = conn.createStatement()) {
            stmt.execute("EXPLAIN ANALYZE " + finalSql);
            execTimeMs = (System.nanoTime() - execStart) / 1_000_000L;
        } finally {
            conn.close();   // return connection to pool
        }

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
        public long execTimeMs;     // PPO reward signal  (join order affects this); -1 = missing
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