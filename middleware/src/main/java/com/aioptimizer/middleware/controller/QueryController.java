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
            System.out.println("2. Extracted SQL String: " + sql);

            if (sql == null || sql.trim().isEmpty()) {
                return ResponseEntity.badRequest().body("ERROR: The SQL query cannot be empty.");
            }

            System.out.println("3. Sending to Firewall...");
            astFirewall.inspectQuery(sql);

            System.out.println("4. Firewall passed! Extracting tables...");
            List<String> tables = extractor.extractTables(sql);

            Map<String, Object> pythonPayload = new HashMap<>();
            pythonPayload.put("tables", tables);
            pythonPayload.put("join_conditions", List.of());
            pythonPayload.put("original_sql", sql);

            String pythonAiUrl = "http://localhost:8000/optimize";
            ResponseEntity<Map> aiResponse = restTemplate.postForEntity(pythonAiUrl, pythonPayload, Map.class);

            String optimizedSql = (String) aiResponse.getBody().get("optimized_query");
            List<String> chosenOrder = (List<String>) aiResponse.getBody().get("choosen_order");

            QueryTelemetryService.TelemetryResult metrics = queryTelemetryService.executeAndTrack(optimizedSql);

            System.out.println("Execution Time: " + metrics.latencyMs + "ms");
            System.out.println("Active Server Connections: " + metrics.activeConnections);

            try {
                Map<String, Object> feedbackPayload = new HashMap<>();
                feedbackPayload.put("tables",             tables);
                feedbackPayload.put("chosen_order",       chosenOrder);
                feedbackPayload.put("latency_ms",         (double) metrics.latencyMs);
                feedbackPayload.put("active_connections", metrics.activeConnections);

                String feedbackUrl = "http://localhost:8000/feedback";
                ResponseEntity<Map> feedbackResponse = restTemplate.postForEntity(feedbackUrl, feedbackPayload, Map.class);

                double reward     = ((Number) feedbackResponse.getBody().get("reward")).doubleValue();
                int    bufferSize = (int)     feedbackResponse.getBody().get("buffer_size");
                System.out.println("PPO Reward: " + reward + " | Replay Buffer: " + bufferSize + "/1000");

            } catch (Exception feedbackEx) {
                System.out.println("Warning: Feedback to AI engine failed: " + feedbackEx.getMessage());
            }

            Map<String, Object> finalResponse = new HashMap<>(aiResponse.getBody());
            finalResponse.put("latency_ms",         metrics.latencyMs);
            finalResponse.put("active_connections", metrics.activeConnections);

            return ResponseEntity.ok(finalResponse);

        } catch (Exception e) {
            e.printStackTrace();
            return ResponseEntity.badRequest().body("ERROR: " + e.getMessage());
        }
    }
}