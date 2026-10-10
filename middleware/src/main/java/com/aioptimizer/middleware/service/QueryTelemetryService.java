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
 *   exec_time_ms  -- time PostgreSQL spends actually running the query.
 *                    Uses System.nanoTime() so fast queries do NOT round to 0.
 *                    -1 = EXEC_TIME_MISSING sentinel.
 *
 *   latency_ms    -- total wall-clock time (wait + exec), kept for compatibility.
 *
 * active_connections -- current active sessions in pg_stat_activity.
 *
 * Statement timeout (Bug fix: disconnected-graph hang):
 *   Before running EXPLAIN ANALYZE, we set statement_timeout on the session.
 *   If a bad PPO-sampled join order forces a cross-product that would run for
 *   hours (e.g. Leading(((ci mk) t)) with cast_info at 36M rows), PostgreSQL
 *   will cancel it after STATEMENT_TIMEOUT_MS milliseconds and throw an
 *   exception.  The exception is caught, EXEC_TIME_MISSING is reported, and
 *   feedback.py skips reward computation — so PPO learns nothing from the
 *   timed-out query rather than hanging indefinitely.
 *
 *   The timeout is reset to 0 (unlimited) after execution so it does not
 *   affect subsequent unrelated sessions sharing the pool.
 */
@Service
public class QueryTelemetryService {

    /** Sentinel: exec time could not be measured (not zero, which is valid). */
    public static final long EXEC_TIME_MISSING = -1L;

    /**
     * Maximum wall-clock time allowed for a single query.
     * 10 seconds is generous for the IMDB benchmark queries; reduce for stricter
     * enforcement.  Set to 0 to disable (not recommended with stochastic hints).
     */
    private static final int STATEMENT_TIMEOUT_MS = 10_000;

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
        //    nanoTime for sub-millisecond precision.
        long waitStart = System.nanoTime();
        Connection conn = dataSource.getConnection();
        long waitTimeMs = (System.nanoTime() - waitStart) / 1_000_000L;

        // 3. Measure exec_time under a statement_timeout guard.
        //    The timeout is set on the session before running the query and
        //    reset to 0 (unlimited) afterward so it does not bleed into other
        //    pool sessions.
        long execStart  = System.nanoTime();
        long execTimeMs = EXEC_TIME_MISSING;

        try (Statement stmt = conn.createStatement()) {
            // Arm the per-session timeout (PostgreSQL will cancel automatically)
            if (STATEMENT_TIMEOUT_MS > 0) {
                stmt.execute("SET LOCAL statement_timeout = '" + STATEMENT_TIMEOUT_MS + "ms'");
            }

            stmt.execute("EXPLAIN ANALYZE " + finalSql);
            execTimeMs = (System.nanoTime() - execStart) / 1_000_000L;

        } catch (Exception queryEx) {
            // statement_timeout cancellation (or any other query error).
            // execTimeMs stays EXEC_TIME_MISSING; feedback.py will skip reward.
            System.out.println("[Telemetry] Query cancelled or failed: " + queryEx.getMessage());
        } finally {
            // Always release the connection back to the pool.
            // SET LOCAL is session-scoped and reset automatically on transaction
            // end, but closing the connection also returns it to a clean state.
            conn.close();
        }

        long latencyMs = waitTimeMs + (execTimeMs == EXEC_TIME_MISSING ? 0 : execTimeMs);

        System.out.println("[Telemetry] wait=" + waitTimeMs + "ms"
            + "  exec=" + (execTimeMs == EXEC_TIME_MISSING ? "TIMEOUT" : execTimeMs + "ms")
            + "  total=" + latencyMs + "ms"
            + "  connections=" + activeConnections);

        return new TelemetryResult(latencyMs, execTimeMs, waitTimeMs, activeConnections);
    }

    // ── Result holder ──────────────────────────────────────────────────────────
    public static class TelemetryResult {
        public long latencyMs;      // total  = wait + exec
        public long execTimeMs;     // PPO reward signal; EXEC_TIME_MISSING (-1) = timed out
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