package com.aioptimizer.middleware.controller;

import com.aioptimizer.middleware.security.AstFirewall;
import com.aioptimizer.middleware.service.QueryFeatureExtractor;
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

    private AstFirewall astFirewall;

    private QueryFeatureExtractor extractor;

    private final RestTemplate restTemplate = new RestTemplate();

    public QueryController(AstFirewall astFirewall, QueryFeatureExtractor extractor) {
        this.astFirewall = astFirewall;
        this.extractor = extractor;
    }

    @PostMapping("/execute")
    public ResponseEntity<?> executeQuery(@RequestBody String sql){
        try {
            astFirewall.inspectQuery(sql);

            List<String> tables = extractor.extractTables(sql);

            Map<String, Object> payload = new HashMap<>();
            payload.put("tables",tables);
            payload.put("join_conditions",List.of());
            payload.put("original_sql",sql);

            String pythonAiUrl = "http://localhost:8000/optimize";
            ResponseEntity<Map> response = restTemplate.postForEntity(pythonAiUrl, payload, Map.class);
            return ResponseEntity.ok(response.getBody());
        } catch (Exception e) {
            return ResponseEntity.badRequest().body("ERROR: "+ e.getMessage());
        }
    }
}
