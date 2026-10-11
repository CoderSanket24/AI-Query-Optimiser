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
 * Statement timeout guard:
 *   SET statement_timeout is issued on the session before every EXPLAIN ANALYZE.
 *   PostgreSQL §20.1: SET LOCAL outside a transaction block has NO effect and
 *   emits a warning.  HikariCP uses autocommit (no open transaction), so we use
 *   plain SET (session-scoped).  An explicit RESET statement_timeout is issued in
 *   a finally block BEFORE conn.close() so the connection is clean when it returns
 *   to the pool.  If RESET fails, Connection.abort() is called so HikariCP evicts
 *   the dirty connection rather than recycling it.
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
        long waitStart = System.nanoTime();
        Connection conn = dataSource.getConnection();
        long waitTimeMs = (System.nanoTime() - waitStart) / 1_000_000L;

        long execStart  = System.nanoTime();
        long execTimeMs = EXEC_TIME_MISSING;

        try {
            // 3. Execute under statement_timeout guard.
            //    SET (session-scoped) works on autocommit connections.
            //    SET LOCAL outside a transaction block has no effect (PostgreSQL §20.1).
            try (Statement stmt = conn.createStatement()) {
                if (STATEMENT_TIMEOUT_MS > 0) {
                    stmt.execute("SET statement_timeout = '" + STATEMENT_TIMEOUT_MS + "ms'");
                }
                stmt.execute("EXPLAIN ANALYZE " + finalSql);
                execTimeMs = (System.nanoTime() - execStart) / 1_000_000L;

            } catch (Exception queryEx) {
                // Timeout (SQLState 57014) or any other query error.
                // execTimeMs stays EXEC_TIME_MISSING; feedback.py skips reward.
                System.out.println("[Telemetry] Query failed/cancelled"
                    + " (SQLState=" + sqlStateOf(queryEx) + "): " + queryEx.getMessage());

            } finally {
                // Always RESET before returning the connection to the pool —
                // whether the query succeeded, timed out, or failed for another reason.
                // A fresh Statement is needed because the prior one was closed by TWR.
                if (STATEMENT_TIMEOUT_MS > 0) {
                    resetTimeout(conn);
                }
            }
        } finally {
            // conn.close() always runs.  If resetTimeout() called abort(), HikariCP
            // detects the dead physical connection here and evicts it instead of
            // recycling it — no separate eviction call needed.
            conn.close();
        }

        long latencyMs = waitTimeMs + (execTimeMs == EXEC_TIME_MISSING ? 0 : execTimeMs);
        System.out.println("[Telemetry] wait=" + waitTimeMs + "ms"
            + "  exec=" + (execTimeMs == EXEC_TIME_MISSING ? "CANCELLED" : execTimeMs + "ms")
            + "  total=" + latencyMs + "ms"
            + "  connections=" + activeConnections);
        return new TelemetryResult(latencyMs, execTimeMs, waitTimeMs, activeConnections);
    }

    /**
     * Runs RESET statement_timeout on a fresh Statement so the pooled connection
     * is clean for the next borrower.  If RESET itself fails (network drop, etc.)
     * the physical connection is aborted; HikariCP will evict it on close().
     */
    private void resetTimeout(Connection conn) {
        try (Statement rst = conn.createStatement()) {
            rst.execute("RESET statement_timeout");
        } catch (Exception resetEx) {
            System.err.println("[Telemetry] RESET statement_timeout failed — "
                + "aborting connection to prevent dirty pool re-use. Cause: "
                + resetEx.getMessage());
            try {
                // abort() closes the underlying physical connection.
                // conn.close() in the outer finally then tells HikariCP the
                // connection is dead, and HikariCP evicts it from the pool.
                conn.abort(Runnable::run);
            } catch (Exception abortEx) {
                // Last-resort: log clearly. The connection will be returned to
                // the pool with statement_timeout still set; HikariCP's
                // connection-validation query (isValid / keepalive) may evict it
                // on the next borrow cycle, but there is no hard guarantee.
                System.err.println("[Telemetry] abort() also failed — connection "
                    + "may be returned dirty. Cause: " + abortEx.getMessage());
            }
        }
    }

    /** Extracts the SQLState string from a SQLException, or "n/a" for other exceptions. */
    private static String sqlStateOf(Exception ex) {
        if (ex instanceof java.sql.SQLException sqle && sqle.getSQLState() != null) {
            return sqle.getSQLState();
        }
        return "n/a";
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