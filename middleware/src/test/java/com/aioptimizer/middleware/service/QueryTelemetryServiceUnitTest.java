package com.aioptimizer.middleware.service;

import org.junit.jupiter.api.Test;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.test.util.ReflectionTestUtils;

import javax.sql.DataSource;
import java.lang.reflect.Proxy;
import java.sql.Connection;
import java.sql.SQLException;
import java.sql.Statement;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit tests for QueryTelemetryService.executeAndTrack().
 *
 * Uses JDK Proxy doubles for Connection/Statement/DataSource so no extra
 * dependencies (Mockito, etc.) are needed beyond JUnit 5 and spring-test.
 *
 * Four scenarios:
 *   1. Normal query  — SET issued, EXPLAIN succeeds, RESET issued, no abort.
 *   2. Query timeout — SET issued, EXPLAIN throws, RESET still issued, no abort.
 *   3. RESET fails   — EXPLAIN succeeds, RESET throws, abort() called.
 *   4. Timeout then RESET fails — both errors, abort() called.
 */
class QueryTelemetryServiceUnitTest {

    // ── Test-double builders ─────────────────────────────────────────────────

    /**
     * Returns a Proxy Statement that appends every sql string to {@code log}.
     * If {@code failContaining} is non-null, throws a SQLException with the
     * given sqlState when that substring appears in the sql.
     */
    private Statement stubStatement(List<String> log,
                                    String failContaining,
                                    String sqlState) {
        return (Statement) Proxy.newProxyInstance(
            getClass().getClassLoader(),
            new Class[]{Statement.class},
            (proxy, method, args) -> {
                if ("execute".equals(method.getName())
                        && args != null && args.length >= 1) {
                    String sql = (String) args[0];
                    log.add(sql);
                    if (failContaining != null && sql.contains(failContaining)) {
                        throw new SQLException("Simulated failure: " + sql, sqlState);
                    }
                    return Boolean.FALSE;
                }
                // isClosed, getWarnings, clearWarnings, close, addBatch, etc.
                Class<?> ret = method.getReturnType();
                if (ret == boolean.class)  return Boolean.FALSE;
                if (ret == int.class)      return 0;
                return null;   // void / reference — fine for close() etc.
            });
    }

    /**
     * Returns a Proxy Connection that hands out {@code stmts} in order from
     * createStatement(). Records abort() and close() calls.
     */
    private Connection stubConnection(AtomicBoolean abortCalled,
                                      AtomicBoolean closeCalled,
                                      Statement... stmts) {
        AtomicInteger idx = new AtomicInteger(0);
        return (Connection) Proxy.newProxyInstance(
            getClass().getClassLoader(),
            new Class[]{Connection.class},
            (proxy, method, args) -> switch (method.getName()) {
                case "createStatement" -> stmts[idx.getAndIncrement()];
                case "abort"           -> { abortCalled.set(true); yield null; }
                case "close"           -> { closeCalled.set(true); yield null; }
                default -> {
                    Class<?> ret = method.getReturnType();
                    yield (ret == boolean.class) ? Boolean.FALSE : null;
                }
            });
    }

    /** Returns a Proxy DataSource whose getConnection() always returns conn. */
    private DataSource stubDataSource(Connection conn) {
        return (DataSource) Proxy.newProxyInstance(
            getClass().getClassLoader(),
            new Class[]{DataSource.class},
            (proxy, method, args) ->
                "getConnection".equals(method.getName()) ? conn : null);
    }

    /**
     * JdbcTemplate subclass that returns a fixed connection count without
     * touching a real database. Only queryForObject(String, Class) is used
     * by executeAndTrack() for the pg_stat_activity count.
     */
    @SuppressWarnings("unchecked")
    private JdbcTemplate fixedCountTemplate() {
        return new JdbcTemplate() {
            @Override
            public <T> T queryForObject(String sql, Class<T> requiredType) {
                return (T) Integer.valueOf(1);   // pretend 1 active connection
            }
        };
    }

    /** Assembles and returns a fully configured service under test. */
    private QueryTelemetryService buildSvc(DataSource ds) {
        QueryTelemetryService svc = new QueryTelemetryService();
        ReflectionTestUtils.setField(svc, "jdbcTemplate", fixedCountTemplate());
        ReflectionTestUtils.setField(svc, "dataSource",   ds);
        return svc;
    }

    // ── Test cases ───────────────────────────────────────────────────────────

    /**
     * Happy path: SET must appear before EXPLAIN ANALYZE, RESET must follow,
     * abort must NOT be called, close must be called exactly once.
     */
    @Test
    void normalQuery_setThenExplainThenReset_noAbort() throws Exception {
        List<String> mainLog  = new ArrayList<>();
        List<String> resetLog = new ArrayList<>();
        AtomicBoolean abortCalled = new AtomicBoolean(false);
        AtomicBoolean closeCalled = new AtomicBoolean(false);

        Statement mainStmt  = stubStatement(mainLog,  null, null);
        Statement resetStmt = stubStatement(resetLog, null, null);
        Connection conn = stubConnection(abortCalled, closeCalled, mainStmt, resetStmt);

        QueryTelemetryService.TelemetryResult result =
            buildSvc(stubDataSource(conn)).executeAndTrack("SELECT 1");

        // First statement block: SET then EXPLAIN
        assertEquals(2, mainLog.size(), "Expected SET + EXPLAIN on the main statement");
        assertTrue(mainLog.get(0).startsWith("SET statement_timeout"),
            "First sql must be SET statement_timeout, got: " + mainLog.get(0));
        assertTrue(mainLog.get(1).startsWith("EXPLAIN ANALYZE"),
            "Second sql must be EXPLAIN ANALYZE, got: " + mainLog.get(1));

        // Reset statement block: exactly one RESET
        assertEquals(1, resetLog.size(), "Expected exactly one RESET");
        assertEquals("RESET statement_timeout", resetLog.get(0));

        // abort must NOT be called on success
        assertFalse(abortCalled.get(), "abort() must not be called when RESET succeeds");

        // conn.close() must be called
        assertTrue(closeCalled.get(), "conn.close() must always be called");

        // exec time must have been recorded (not EXEC_TIME_MISSING)
        assertNotEquals(QueryTelemetryService.EXEC_TIME_MISSING, result.execTimeMs,
            "Normal query must record a real exec time");
    }

    /**
     * Timeout path: EXPLAIN throws SQLState 57014.  RESET must still be issued
     * (the connection is valid after a cancelled statement), abort must not fire,
     * and execTimeMs must be EXEC_TIME_MISSING.
     */
    @Test
    void cancelledQuery_resetStillIssued_execTimeMissing() throws Exception {
        List<String> mainLog  = new ArrayList<>();
        List<String> resetLog = new ArrayList<>();
        AtomicBoolean abortCalled = new AtomicBoolean(false);
        AtomicBoolean closeCalled = new AtomicBoolean(false);

        // EXPLAIN ANALYZE is the string that will trigger the throw
        Statement mainStmt  = stubStatement(mainLog,  "EXPLAIN ANALYZE", "57014");
        Statement resetStmt = stubStatement(resetLog, null, null);
        Connection conn = stubConnection(abortCalled, closeCalled, mainStmt, resetStmt);

        QueryTelemetryService.TelemetryResult result =
            buildSvc(stubDataSource(conn)).executeAndTrack("SELECT pg_sleep(30)");

        assertTrue(mainLog.get(0).startsWith("SET statement_timeout"));
        assertTrue(mainLog.get(1).startsWith("EXPLAIN ANALYZE"));
        assertEquals(1, resetLog.size());
        assertEquals("RESET statement_timeout", resetLog.get(0));

        assertFalse(abortCalled.get(), "abort() must not be called when RESET succeeds");
        assertTrue(closeCalled.get());
        assertEquals(QueryTelemetryService.EXEC_TIME_MISSING, result.execTimeMs,
            "Cancelled query must report EXEC_TIME_MISSING");
    }

    /**
     * RESET fails (e.g. network drop after the query finishes): abort() must be
     * called so HikariCP evicts the dirty connection; close() must still be called.
     */
    @Test
    void resetFails_abortCalled_closeStillCalled() throws Exception {
        List<String> mainLog  = new ArrayList<>();
        List<String> resetLog = new ArrayList<>();
        AtomicBoolean abortCalled = new AtomicBoolean(false);
        AtomicBoolean closeCalled = new AtomicBoolean(false);

        Statement mainStmt  = stubStatement(mainLog,  null,    null);
        Statement resetStmt = stubStatement(resetLog, "RESET", "08006"); // connection_failure

        Connection conn = stubConnection(abortCalled, closeCalled, mainStmt, resetStmt);

        // executeAndTrack must not throw even though RESET failed
        assertDoesNotThrow(() -> buildSvc(stubDataSource(conn)).executeAndTrack("SELECT 1"));

        assertEquals("RESET statement_timeout", resetLog.get(0));
        assertTrue(abortCalled.get(), "abort() must be called when RESET fails");
        assertTrue(closeCalled.get(), "close() must be called even after abort()");
    }

    /**
     * Worst case: EXPLAIN times out AND RESET fails.  Both paths must be handled;
     * abort() must still fire; execTimeMs must be EXEC_TIME_MISSING.
     */
    @Test
    void timeoutThenResetFails_abortCalled_execTimeMissing() throws Exception {
        List<String> mainLog  = new ArrayList<>();
        List<String> resetLog = new ArrayList<>();
        AtomicBoolean abortCalled = new AtomicBoolean(false);
        AtomicBoolean closeCalled = new AtomicBoolean(false);

        Statement mainStmt  = stubStatement(mainLog,  "EXPLAIN ANALYZE", "57014");
        Statement resetStmt = stubStatement(resetLog, "RESET",           "08006");
        Connection conn = stubConnection(abortCalled, closeCalled, mainStmt, resetStmt);

        QueryTelemetryService.TelemetryResult result =
            buildSvc(stubDataSource(conn)).executeAndTrack("SELECT pg_sleep(30)");

        assertTrue(abortCalled.get());
        assertTrue(closeCalled.get());
        assertEquals(QueryTelemetryService.EXEC_TIME_MISSING, result.execTimeMs);
    }
}
