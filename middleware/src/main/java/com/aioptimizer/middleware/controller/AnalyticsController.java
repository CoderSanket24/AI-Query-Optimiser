package com.aioptimizer.middleware.controller;

import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.client.RestTemplate;

import java.util.Map;

/**
 * AnalyticsController
 * -------------------
 * Proxies analytics + anomaly endpoints from FastAPI (:8000) through
 * Spring Boot (:8080) so the React frontend NEVER talks to FastAPI directly.
 *
 * All database services connect to Spring Boot only.
 * FastAPI is purely the AI engine — not accessible from the browser.
 *
 * Endpoints proxied:
 *   GET /api/analytics/summary     -> FastAPI /analytics/summary
 *   GET /api/analytics/history     -> FastAPI /analytics/history
 *   GET /api/analytics/rewards     -> FastAPI /analytics/rewards
 *   GET /api/analytics/ppo/stats   -> FastAPI /feedback/stats
 *   GET /api/analytics/anomaly     -> FastAPI /anomaly/status
 */
@RestController
@RequestMapping("/api/analytics")
public class AnalyticsController {

    private final RestTemplate restTemplate = new RestTemplate();
    private static final String FASTAPI_BASE = "http://localhost:8000";

    @GetMapping("/summary")
    public ResponseEntity<?> getSummary() {
        try {
            ResponseEntity<Map> res = restTemplate.getForEntity(FASTAPI_BASE + "/analytics/summary", Map.class);
            return ResponseEntity.ok(res.getBody());
        } catch (Exception e) {
            return ResponseEntity.status(503).body("FastAPI unavailable: " + e.getMessage());
        }
    }

    @GetMapping("/history")
    public ResponseEntity<?> getHistory() {
        try {
            ResponseEntity<Map> res = restTemplate.getForEntity(FASTAPI_BASE + "/analytics/history", Map.class);
            return ResponseEntity.ok(res.getBody());
        } catch (Exception e) {
            return ResponseEntity.status(503).body("FastAPI unavailable: " + e.getMessage());
        }
    }

    @GetMapping("/rewards")
    public ResponseEntity<?> getRewards() {
        try {
            ResponseEntity<Map> res = restTemplate.getForEntity(FASTAPI_BASE + "/analytics/rewards", Map.class);
            return ResponseEntity.ok(res.getBody());
        } catch (Exception e) {
            return ResponseEntity.status(503).body("FastAPI unavailable: " + e.getMessage());
        }
    }

    @GetMapping("/ppo/stats")
    public ResponseEntity<?> getPpoStats() {
        try {
            ResponseEntity<Map> res = restTemplate.getForEntity(FASTAPI_BASE + "/feedback/stats", Map.class);
            return ResponseEntity.ok(res.getBody());
        } catch (Exception e) {
            return ResponseEntity.status(503).body("FastAPI unavailable: " + e.getMessage());
        }
    }

    @GetMapping("/anomaly")
    public ResponseEntity<?> getAnomalyStatus() {
        try {
            ResponseEntity<Map> res = restTemplate.getForEntity(FASTAPI_BASE + "/anomaly/status", Map.class);
            return ResponseEntity.ok(res.getBody());
        } catch (Exception e) {
            return ResponseEntity.status(503).body("FastAPI unavailable: " + e.getMessage());
        }
    }
}
