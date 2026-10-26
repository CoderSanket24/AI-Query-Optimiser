package com.aioptimizer.middleware.service;

import com.zaxxer.hikari.HikariConfig;
import com.zaxxer.hikari.HikariDataSource;
import org.junit.jupiter.api.*;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.test.util.ReflectionTestUtils;

import java.sql.Connection;
import java.sql.ResultSet;
import java.sql.Statement;
import java.util.TimeZone;

import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * Integration tests for QueryTelemetryService — require a live PostgreSQL instance.
 *
 * Gate: set the JOINSENSE_TEST_DB_URL environment variable to a valid JDBC URL
 * (e.g. "jdbc:postgresql://localhost:5432/imdb").  If the variable is absent the
 * entire class is skipped (JUnit Assumption).
 *
 * Optional variables (fall back to sensible defaults):
 *   JOINSENSE_TEST_DB_USER     (default: "postgres")
 *   JOINSENSE_TEST_DB_PASSWORD (default: "")
 *
 * Three acceptance criteria from item 1.1:
 *   1. SELECT pg_sleep(120) is cancelled near STATEMENT_TIMEOUT_MS — not after 120 s.
 *   2. A normal query (SELECT 1) completes with a recorded exec time.
 *   3. After executeAndTrack() returns, the pooled connection's statement_timeout
 *      is back to the server's configured default (verified via SHOW statement_timeout).
 *
 * The pool is configured with maximumPoolSize=1 so HikariCP always reuses the same
 * physical connection, making assertion 3 deterministic.
 */
@TestInstance(TestInstance.Lifecycle.PER_CLASS)
@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
class QueryTelemetryServiceIntegrationTest {

    private HikariDataSource  ds;
    private QueryTelemetryService svc;

    /** The value of SHOW statement_timeout captured before any test touches it. */
    private String initialTimeout;

    @BeforeAll
    void setup() throws Exception {
        assumeTrue(
            System.getenv("JOINSENSE_TEST_DB_URL") != null,
            "JOINSENSE_TEST_DB_URL not set — skipping integration tests"
        );

        // PostgreSQL 18 rejects "Asia/Calcutta" (deprecated alias used by the Windows JVM).
        // Set the JVM timezone to UTC before the pool opens so the driver sends a valid name.
        // This affects only this test JVM, not statement_timeout behaviour.
        TimeZone.setDefault(TimeZone.getTimeZone("UTC"));

        HikariConfig cfg = new HikariConfig();
        cfg.setJdbcUrl(System.getenv("JOINSENSE_TEST_DB_URL"));
        cfg.setUsername(System.getenv().getOrDefault("JOINSENSE_TEST_DB_USER",     "postgres"));
        cfg.setPassword(System.getenv().getOrDefault("JOINSENSE_TEST_DB_PASSWORD", ""));
        // Pool size 1: guarantees we always borrow the same physical connection,
        // making the "timeout restored" assertion reliable.
        cfg.setMaximumPoolSize(1);
        ds = new HikariDataSource(cfg);

        // Capture the server's default so we can compare after each test.
        // Typically '0' (disabled) unless postgresql.conf or ALTER ROLE changed it.
        try (Connection c  = ds.getConnection();
             Statement  s  = c.createStatement();
             ResultSet  rs = s.executeQuery("SHOW statement_timeout")) {
            rs.next();
            initialTimeout = rs.getString(1);
        }

        JdbcTemplate template = new JdbcTemplate(ds);
        svc = new QueryTelemetryService();
        ReflectionTestUtils.setField(svc, "jdbcTemplate", template);
        ReflectionTestUtils.setField(svc, "dataSource",   ds);

        System.out.println("[IT] Connected to: " + System.getenv("JOINSENSE_TEST_DB_URL"));
        System.out.println("[IT] Initial statement_timeout: '" + initialTimeout + "'");
        System.out.println("[IT] STATEMENT_TIMEOUT_MS: " + QueryTelemetryService.STATEMENT_TIMEOUT_MS);
    }

    /**
     * Acceptance criterion 1: pg_sleep(120) must be cancelled near
     * STATEMENT_TIMEOUT_MS — not after 120 s.
     * execTimeMs must be EXEC_TIME_MISSING, and the whole call must return well
     * within the sleep duration.
     */
    @Test
    @Order(1)
    void pgSleepIsCancelledAtConfiguredTimeout() {
        long startNs = System.nanoTime();

        // executeAndTrack must NOT throw — it logs and returns EXEC_TIME_MISSING
        QueryTelemetryService.TelemetryResult result =
            assertDoesNotThrow(() -> svc.executeAndTrack("SELECT pg_sleep(120)"),
                "executeAndTrack must not propagate the cancellation exception");

        long elapsedMs = (System.nanoTime() - startNs) / 1_000_000L;
        System.out.println("[IT] pg_sleep(120) elapsed: " + elapsedMs + " ms");

        assertEquals(QueryTelemetryService.EXEC_TIME_MISSING, result.execTimeMs,
            "Cancelled query must report EXEC_TIME_MISSING (-1)");

        // Must complete well before the 120-second sleep duration.
        // Allow 1.5× STATEMENT_TIMEOUT_MS for cancellation and cleanup overhead.
        long maxAcceptableMs = (long) (QueryTelemetryService.STATEMENT_TIMEOUT_MS * 1.5);
        assertTrue(elapsedMs < maxAcceptableMs,
            "Call took " + elapsedMs + " ms; expected < " + maxAcceptableMs
            + " ms (1.5× STATEMENT_TIMEOUT_MS)");
    }

    /**
     * Acceptance criterion 2: a normal query (SELECT 1) returns a real exec time.
     */
    @Test
    @Order(2)
    void normalSelectOneCompletes() {
        QueryTelemetryService.TelemetryResult result =
            assertDoesNotThrow(() -> svc.executeAndTrack("SELECT 1"),
                "SELECT 1 must not throw");

        assertNotEquals(QueryTelemetryService.EXEC_TIME_MISSING, result.execTimeMs,
            "Normal query must record a real exec time, not EXEC_TIME_MISSING");
        assertTrue(result.execTimeMs >= 0,
            "exec time must be non-negative, got: " + result.execTimeMs);

        System.out.println("[IT] SELECT 1 exec time: " + result.execTimeMs + " ms");
    }

    /**
     * Acceptance criterion 3: after executeAndTrack() returns and the connection
     * is back in the pool, SHOW statement_timeout must equal the pre-test value.
     *
     * Because maximumPoolSize=1, this borrow is guaranteed to get the same physical
     * connection that executeAndTrack() just used.
     */
    @Test
    @Order(3)
    void statementTimeoutRestoredOnRecycledPoolConnection() throws Exception {
        try (Connection c  = ds.getConnection();
             Statement  s  = c.createStatement();
             ResultSet  rs = s.executeQuery("SHOW statement_timeout")) {
            rs.next();
            String actual = rs.getString(1);
            System.out.println("[IT] statement_timeout after pool recycle: '" + actual + "'");
            assertEquals(initialTimeout, actual,
                "RESET statement_timeout must restore the server default. "
                + "Expected '" + initialTimeout + "', got '" + actual + "'.");
        }
    }

    @AfterAll
    void teardown() {
        if (ds != null && !ds.isClosed()) {
            ds.close();
        }
    }
}
