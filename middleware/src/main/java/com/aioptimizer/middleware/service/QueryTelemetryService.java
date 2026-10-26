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
 *                    -1 = EXEC_TIME_MISSING sentinel. Starts immediately before
 *                    EXPLAIN ANALYZE; explain_call_ms is a cross-check.
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
 *
 * Per-stage latency instrumentation (fractional ms, double):
 *   active_conn_query_ms -- time for pg_stat_activity count query
 *   pool_wait_ms         -- same as waitTimeMs (int→double alias for breakdown)
 *   set_timeout_ms       -- time to execute SET statement_timeout
 *   blks_before_query_ms -- time to snapshot pg_stat_database.blks_read before query
 *   explain_call_ms      -- time of the EXPLAIN ANALYZE call only (recorded on cancel)
 *   blks_after_query_ms  -- time to snapshot pg_stat_database.blks_read after query
 *   reset_ms             -- time to execute RESET statement_timeout
 *   WARNING logged if blks_before or blks_after query takes > 100 ms.
 */
@Service
public class QueryTelemetryService {

    /** Sentinel: exec time could not be measured (not zero, which is valid). */
    public static final long EXEC_TIME_MISSING = -1L;

    /**
     * Maximum wall-clock time allowed for a single query, in milliseconds.
     * STATEMENT_TIMEOUT_MS is generous for the IMDB benchmark queries; reduce
     * for stricter enforcement.  Set to 0 to disable (not recommended with
     * stochastic hints).
     */
    public static final int STATEMENT_TIMEOUT_MS = 50_000;

    @Autowired
    private JdbcTemplate jdbcTemplate;

    @Autowired
    private DataSource dataSource;

    public TelemetryResult executeAndTrack(String finalSql) throws Exception {

        // 1. Count active connections BEFORE execution (baseline load snapshot).
        //    Timed separately so the cost of this query is visible in the breakdown.
        String contentionQuery =
            "SELECT count(*) FROM pg_stat_activity WHERE state = 'active';";
        long _acqT0 = System.nanoTime();
        Integer activeConnections =
            jdbcTemplate.queryForObject(contentionQuery, Integer.class);
        double activeConnQueryMs = (System.nanoTime() - _acqT0) / 1e6;

        // 2. Measure pool_wait_ms: time to acquire connection from HikariCP pool.
        long waitStart = System.nanoTime();
        Connection conn = dataSource.getConnection();
        long waitTimeMs = (System.nanoTime() - waitStart) / 1_000_000L;

        long execTimeMs     = EXEC_TIME_MISSING;
        long blksReadBefore = 0L;
        long blksReadAfter  = 0L;

        // Per-stage timer accumulators (fractional ms, set onto result after construction).
        double setTimeoutMs      = 0.0;
        double explainCallMs     = 0.0;
        double blksBeforeQueryMs = 0.0;
        double blksAfterQueryMs  = 0.0;
        double resetMs           = 0.0;

        try {
            // 3. Execute under statement_timeout guard.
            //    SET (session-scoped) works on autocommit connections.
            //    SET LOCAL outside a transaction block has no effect (PostgreSQL §20.1).
            try (Statement stmt = conn.createStatement()) {

                // SET statement_timeout — timed separately from EXPLAIN ANALYZE.
                if (STATEMENT_TIMEOUT_MS > 0) {
                    long _st0 = System.nanoTime();
                    stmt.execute("SET statement_timeout = '" + STATEMENT_TIMEOUT_MS + "ms'");
                    setTimeoutMs = (System.nanoTime() - _st0) / 1e6;
                }

                // Snapshot I/O counters before the query runs.
                long _bb0 = System.nanoTime();
                blksReadBefore = queryBlksRead(conn);
                blksBeforeQueryMs = (System.nanoTime() - _bb0) / 1e6;
                if (blksBeforeQueryMs > 100.0) {
                    System.out.printf(
                        "[Telemetry] WARNING: blks_before_query_ms=%.1f ms (> 100 ms threshold)%n",
                        blksBeforeQueryMs);
                }

                // EXPLAIN ANALYZE — timed via its own nested try/finally so that
                // explain_call_ms is recorded even when the query is cancelled.
                // execTimeMs stays EXEC_TIME_MISSING on cancellation.
                long _explainStart = System.nanoTime();
                try {
                    long execStart = System.nanoTime();
                    stmt.execute("EXPLAIN ANALYZE " + finalSql);
                    execTimeMs = (System.nanoTime() - execStart) / 1_000_000L;
                } catch (Exception queryEx) {
                    // Timeout (SQLState 57014) or any other query error.
                    // execTimeMs stays EXEC_TIME_MISSING; feedback.py skips reward.
                    System.out.println("[Telemetry] Query failed/cancelled"
                        + " (SQLState=" + sqlStateOf(queryEx) + "): " + queryEx.getMessage());
                } finally {
                    // Always captured — gives the actual EXPLAIN wall-clock even on cancel.
                    explainCallMs = (System.nanoTime() - _explainStart) / 1e6;
                }

            } catch (Exception outerEx) {
                // Catches SET statement_timeout failures or blks_before query errors.
                // EXPLAIN errors are caught by the inner try/catch above.
                System.out.println("[Telemetry] Query execution error"
                    + " (SQLState=" + sqlStateOf(outerEx) + "): " + outerEx.getMessage());

            } finally {
                // Snapshot I/O counters after the query finishes (or is cancelled).
                long _ba0 = System.nanoTime();
                blksReadAfter = queryBlksRead(conn);
                blksAfterQueryMs = (System.nanoTime() - _ba0) / 1e6;
                if (blksAfterQueryMs > 100.0) {
                    System.out.printf(
                        "[Telemetry] WARNING: blks_after_query_ms=%.1f ms (> 100 ms threshold)%n",
                        blksAfterQueryMs);
                }

                // Always RESET before returning the connection to the pool —
                // whether the query succeeded, timed out, or failed for another reason.
                // A fresh Statement is needed because the prior one was closed by TWR.
                if (STATEMENT_TIMEOUT_MS > 0) {
                    long _rt0 = System.nanoTime();
                    resetTimeout(conn);
                    resetMs = (System.nanoTime() - _rt0) / 1e6;
                }
            }
        } finally {
            // conn.close() always runs.  If resetTimeout() called abort(), HikariCP
            // detects the dead physical connection here and evicts it instead of
            // recycling it — no separate eviction call needed.
            conn.close();
        }

        long latencyMs = waitTimeMs + (execTimeMs == EXEC_TIME_MISSING ? 0 : execTimeMs);
        System.out.printf(
            "[Telemetry] wait=%dms  exec=%s  explain_call=%.1fms  total=%dms"
            + "  conn=%d  blks_delta=%d%n",
            waitTimeMs,
            execTimeMs == EXEC_TIME_MISSING ? "CANCELLED" : execTimeMs + "ms",
            explainCallMs,
            latencyMs,
            activeConnections,
            blksReadAfter - blksReadBefore);

        // Constructor signature is unchanged — new fields are set as public mutables.
        TelemetryResult result = new TelemetryResult(latencyMs, execTimeMs, waitTimeMs,
                                                     activeConnections, blksReadBefore, blksReadAfter);
        result.activeConnQueryMs = activeConnQueryMs;
        result.setTimeoutMs      = setTimeoutMs;
        result.explainCallMs     = explainCallMs;
        result.blksBeforeQueryMs = blksBeforeQueryMs;
        result.blksAfterQueryMs  = blksAfterQueryMs;
        result.resetMs           = resetMs;
        return result;
    }

    /**
     * Reads the cumulative blks_read counter for this database from pg_stat_database.
     * Returns 0 on any error so a stat failure never interrupts query execution.
     */
    private long queryBlksRead(Connection conn) {
        try (Statement stmt = conn.createStatement()) {
            String statSql =
                "SELECT blks_read FROM pg_stat_database WHERE datname = current_database()";
            try (java.sql.ResultSet rs = stmt.executeQuery(statSql)) {
                return rs.next() ? rs.getLong(1) : 0L;
            }
        } catch (Exception e) {
            return 0L;
        }
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
        /** pg_stat_database.blks_read snapshot taken immediately before query execution. */
        public long blksReadBefore;
        /** pg_stat_database.blks_read snapshot taken immediately after query execution. */
        public long blksReadAfter;

        // Per-stage latency fields (fractional ms). Set by executeAndTrack() after
        // construction so the constructor signature remains unchanged.
        /** Time to query pg_stat_activity for active connection count. */
        public double activeConnQueryMs;
        /** Time to execute SET statement_timeout (session-scoped). */
        public double setTimeoutMs;
        /**
         * Wall-clock time of the EXPLAIN ANALYZE call only — recorded even when the
         * query is cancelled.  It serves as a cross-check for execTimeMs.
         */
        public double explainCallMs;
        /** Time to query pg_stat_database.blks_read before EXPLAIN ANALYZE. */
        public double blksBeforeQueryMs;
        /** Time to query pg_stat_database.blks_read after EXPLAIN ANALYZE (or cancel). */
        public double blksAfterQueryMs;
        /** Time to execute RESET statement_timeout on the recycled connection. */
        public double resetMs;

        public TelemetryResult(long latencyMs, long execTimeMs,
                               long waitTimeMs, int activeConnections,
                               long blksReadBefore, long blksReadAfter) {
            this.latencyMs         = latencyMs;
            this.execTimeMs        = execTimeMs;
            this.waitTimeMs        = waitTimeMs;
            this.activeConnections = activeConnections;
            this.blksReadBefore    = blksReadBefore;
            this.blksReadAfter     = blksReadAfter;
        }
    }
}
