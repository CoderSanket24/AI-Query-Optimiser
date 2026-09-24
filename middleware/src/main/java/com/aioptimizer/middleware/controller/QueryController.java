package com.aioptimizer.middleware.controller;

import com.aioptimizer.middleware.security.AstFirewall;
import com.aioptimizer.middleware.service.QueryFeatureExtractor;
import com.aioptimizer.middleware.service.QueryTelemetryService;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.client.RestTemplate;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

@RestController
@RequestMapping("api/query")
public class QueryController {

    private QueryTelemetryService queryTelemetryService;
    private AstFirewall astFirewall;
    private QueryFeatureExtractor extractor;

    private final RestTemplate restTemplate = new RestTemplate();

    public QueryController(QueryTelemetryService queryTelemetryService,
                           AstFirewall astFirewall,
                           QueryFeatureExtractor extractor) {
        this.queryTelemetryService = queryTelemetryService;
        this.astFirewall = astFirewall;
        this.extractor = extractor;
    }

    @PostMapping("/execute")
    public ResponseEntity<?> executeQuery(@RequestBody Map<String, String> payload) {
        try {
            System.out.println("1. Raw Payload: " + payload);

            if (payload == null || !payload.containsKey("query")) {
                return ResponseEntity.badRequest().body("ERROR: JSON body must contain a 'query' key.");
            }

            String sql = payload.get("query");
            System.out.println("2. Extracted SQL: " + sql);

            if (sql == null || sql.trim().isEmpty()) {
                return ResponseEntity.badRequest().body("ERROR: SQL query cannot be empty.");
            }

            // Step 3: Firewall check
            System.out.println("3. Sending to Firewall...");
            astFirewall.inspectQuery(sql);

            // Step 4: Extract tables
            System.out.println("4. Firewall passed! Extracting tables...");
            List<String> tables = extractor.extractTables(sql);

            // Step 5: Ask AI engine to optimise join order
            Map<String, Object> pythonPayload = new HashMap<>();
            pythonPayload.put("tables", tables);
            pythonPayload.put("join_conditions", List.of());
            pythonPayload.put("original_sql", sql);

            String pythonAiUrl = "http://localhost:8000/optimize";
            ResponseEntity<Map> aiResponse = restTemplate.postForEntity(pythonAiUrl, pythonPayload, Map.class);

            String       optimizedSql = (String)       aiResponse.getBody().get("optimized_query");
            List<String> chosenOrder  = (List<String>) aiResponse.getBody().get("choosen_order");

            // Extract query_id and log_prob_old for analytics + PPO batch
            Object queryIdObj    = aiResponse.getBody().get("query_id");
            Object logProbOldObj = aiResponse.getBody().get("log_prob_old");
            Integer queryId      = (queryIdObj    instanceof Number) ? ((Number) queryIdObj).intValue()    : null;
            Double  logProbOld   = (logProbOldObj instanceof Number) ? ((Number) logProbOldObj).doubleValue() : null;

            // Step 6: Execute query — measures wait_time_ms AND exec_time_ms separately
            QueryTelemetryService.TelemetryResult metrics = queryTelemetryService.executeAndTrack(optimizedSql);

            System.out.println("Wait Time:    " + metrics.waitTimeMs + " ms");
            System.out.println("Exec Time:    " + metrics.execTimeMs + " ms");
            System.out.println("Total:        " + metrics.latencyMs  + " ms");
            System.out.println("Connections:  " + metrics.activeConnections);

            // Step 7: Send feedback with full telemetry (Isolation Forest + PPO reward)
            try {
                Map<String, Object> feedbackPayload = new HashMap<>();
                feedbackPayload.put("tables",             tables);
                feedbackPayload.put("chosen_order",       chosenOrder);
                feedbackPayload.put("latency_ms",         (double) metrics.latencyMs);
                feedbackPayload.put("exec_time_ms",       (double) metrics.execTimeMs);   // PPO reward
                feedbackPayload.put("wait_time_ms",       (double) metrics.waitTimeMs);   // Anomaly detection
                feedbackPayload.put("active_connections", metrics.activeConnections);
                if (queryId    != null) feedbackPayload.put("query_id",     queryId);
                if (logProbOld != null) feedbackPayload.put("log_prob_old", logProbOld);

                String feedbackUrl = "http://localhost:8000/feedback";
                ResponseEntity<Map> feedbackResponse = restTemplate.postForEntity(feedbackUrl, feedbackPayload, Map.class);

                Object reward     = feedbackResponse.getBody().get("reward");
                Object bufferSize = feedbackResponse.getBody().get("buffer_size");
                Object batchFired = feedbackResponse.getBody().get("ppo_batch_fired");
                Object anomalyDet = feedbackResponse.getBody().get("anomaly_detected");
                System.out.println("PPO Reward: " + reward
                    + " | Buffer: " + bufferSize + "/1000"
                    + " | BatchFired: " + batchFired
                    + " | AnomalyDetected: " + anomalyDet);

            } catch (Exception feedbackEx) {
                System.out.println("Warning: Feedback failed: " + feedbackEx.getMessage());
            }

            // Step 8: Return enriched response
            Map<String, Object> finalResponse = new HashMap<>(aiResponse.getBody());
            finalResponse.put("latency_ms",         metrics.latencyMs);
            finalResponse.put("exec_time_ms",        metrics.execTimeMs);
            finalResponse.put("wait_time_ms",        metrics.waitTimeMs);
            finalResponse.put("active_connections",  metrics.activeConnections);

            return ResponseEntity.ok(finalResponse);

        } catch (Exception e) {
            e.printStackTrace();
            return ResponseEntity.badRequest().body("ERROR: " + e.getMessage());
        }
    }
}