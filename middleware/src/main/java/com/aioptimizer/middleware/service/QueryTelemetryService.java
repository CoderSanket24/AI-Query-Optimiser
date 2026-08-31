package com.aioptimizer.middleware.service;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.stereotype.Service;

@Service
public class QueryTelemetryService {

    @Autowired
    private JdbcTemplate jdbcTemplate;

    public TelemetryResult executeAndTrack(String finalSql) {
        // 1. Check current server contention (active connections)
        String contentionQuery = "SELECT count(*) FROM pg_stat_activity WHERE state = 'active';";
        Integer activeConnections = jdbcTemplate.queryForObject(contentionQuery, Integer.class);

        String analyzeSql = "EXPLAIN ANALYZE " + finalSql;

        // 2. Execute the AI-optimized query and measure latency
        long startTime = System.currentTimeMillis();
        jdbcTemplate.execute(analyzeSql);
        long endTime = System.currentTimeMillis();

        long executionLatency = endTime - startTime;

        return new TelemetryResult(executionLatency, activeConnections);
    }

    // Inner class to hold the results
    public static class TelemetryResult {
        public long latencyMs;
        public int activeConnections;

        public TelemetryResult(long latencyMs, int activeConnections) {
            this.latencyMs = latencyMs;
            this.activeConnections = activeConnections;
        }
    }
}