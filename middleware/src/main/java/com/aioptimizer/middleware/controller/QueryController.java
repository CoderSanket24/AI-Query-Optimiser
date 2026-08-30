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

    public QueryController(QueryTelemetryService queryTelemetryService, AstFirewall astFirewall, QueryFeatureExtractor extractor) {
        this.queryTelemetryService = queryTelemetryService;
        this.astFirewall = astFirewall;
        this.extractor = extractor;
    }

    @PostMapping("/execute")
    public ResponseEntity<?> executeQuery(@RequestBody Map<String, String> payload){
        try {
            System.out.println("1. Raw Payload: " + payload);
            // 1. Check if the JSON payload is missing the key
            if (payload == null || !payload.containsKey("query")) {
                return ResponseEntity.badRequest().body("ERROR: JSON body must contain a 'query' key.");
            }

            String sql = payload.get("query");
            System.out.println("2. Extracted SQL String: " + sql);

            // 2. THE MISSING NULL CHECK: Ensure the string itself isn't empty
            if (sql == null || sql.trim().isEmpty()) {
                System.out.println("FAIL: SQL string is empty or null.");
                return ResponseEntity.badRequest().body("ERROR: The SQL query cannot be empty.");
            }

            System.out.println("3. Null checks passed! Sending to Firewall...");
            astFirewall.inspectQuery(sql);

            System.out.println("4. Firewall passed! Extracting tables...");
            List<String> tables = extractor.extractTables(sql);

            // 5. Build Python payload
            Map<String, Object> pythonPayload = new HashMap<>();
            pythonPayload.put("tables", tables);
            pythonPayload.put("join_conditions", List.of());
            pythonPayload.put("original_sql", sql);

            // 6. Send to Python
            String pythonAiUrl = "http://localhost:8000/optimize";
            ResponseEntity<Map> response = restTemplate.postForEntity(pythonAiUrl, pythonPayload, Map.class);

            // 7. Extract the optimized SQL from Python's response
            String optimizedSql = (String) response.getBody().get("optimized_sql");

            // 8. Run the query and track contention
            QueryTelemetryService.TelemetryResult metrics = queryTelemetryService.executeAndTrack(optimizedSql);

            System.out.println("Execution Time: " + metrics.latencyMs + "ms");
            System.out.println("Active Server Connections: " + metrics.activeConnections);

            return ResponseEntity.ok(response.getBody());
        } catch (Exception e) {
            e.printStackTrace();
            return ResponseEntity.badRequest().body("ERROR: "+ e.getMessage());
        }
    }
}
