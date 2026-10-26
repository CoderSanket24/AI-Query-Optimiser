package com.aioptimizer.middleware.controller;

import com.aioptimizer.middleware.security.AstFirewall;
import com.aioptimizer.middleware.service.CarbonEstimator;
import com.aioptimizer.middleware.service.QueryFeatureExtractor;
import com.aioptimizer.middleware.service.QueryTelemetryService;
import com.aioptimizer.middleware.model.CarbonResult;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.client.RestTemplate;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

@RestController
@RequestMapping("api/query")
public class QueryController {

    /**
     * Safe alias pattern: must start with a letter or underscore, followed by
     * letters, digits, or underscores only.  Quoted aliases containing *&#47; or ;
     * could otherwise escape the pg_hint_plan comment and inject SQL.
     */
    private static final Pattern SAFE_ALIAS = Pattern.compile("^[A-Za-z_][A-Za-z0-9_]*$");

    private QueryTelemetryService queryTelemetryService;
    private AstFirewall           astFirewall;
    private QueryFeatureExtractor extractor;
    private CarbonEstimator       carbonEstimator;

    private final RestTemplate restTemplate = new RestTemplate();

    public QueryController(QueryTelemetryService queryTelemetryService,
                           AstFirewall astFirewall,
                           QueryFeatureExtractor extractor,
                           CarbonEstimator carbonEstimator) {
        this.queryTelemetryService = queryTelemetryService;
        this.astFirewall           = astFirewall;
        this.extractor             = extractor;
        this.carbonEstimator       = carbonEstimator;
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

            // Total-latency timer — starts after input validation, covers all business logic.
            long _totalStart = System.nanoTime();

            // Step 3: Firewall check on the ORIGINAL SQL
            System.out.println("3. Sending to Firewall...");
            long _fw0 = System.nanoTime();
            astFirewall.inspectQuery(sql);
            double firewallMs = (System.nanoTime() - _fw0) / 1e6;

            // Step 4: Extract alias→realName pairs AND join edges from AST.
            //   alias    = identifier pg_hint_plan sees in the SQL  (used for hints)
            //   realName = physical table name                       (used for schema stats)
            //   edges    = list of "leftAlias,rightAlias" strings   (join graph for Python)
            System.out.println("4. Firewall passed! Extracting table refs and join edges...");
            long _ex0 = System.nanoTime();
            QueryFeatureExtractor.ExtractionResult extraction = extractor.extract(sql);
            double extractMs = (System.nanoTime() - _ex0) / 1e6;

            List<String> aliases   = extraction.tableRefs.stream()
                                                          .map(r -> r.alias)
                                                          .collect(Collectors.toList());
            List<String> realNames = extraction.tableRefs.stream()
                                                          .map(r -> r.realName)
                                                          .collect(Collectors.toList());

            // Encode each join edge as "leftAlias,rightAlias" for Python
            List<String> joinConditions = extraction.joinEdges.stream()
                                                              .map(e -> e.leftAlias + "," + e.rightAlias)
                                                              .collect(Collectors.toList());

            System.out.println("Aliases        : " + aliases);
            System.out.println("RealNames      : " + realNames);
            System.out.println("Join edges     : " + joinConditions);

            // Step 4b: Alias whitelist — every alias must match ^[A-Za-z_][A-Za-z0-9_]*$.
            // A quoted alias like `*/ ; DROP TABLE` could escape the pg_hint_plan comment
            // and inject SQL into the executed string.  Flag unsafe aliases and tell Python
            // to run the query unhinted.
            boolean aliasesAreSafe = aliases.stream().allMatch(a -> SAFE_ALIAS.matcher(a).matches());
            if (!aliasesAreSafe) {
                List<String> unsafe = aliases.stream()
                                             .filter(a -> !SAFE_ALIAS.matcher(a).matches())
                                             .collect(Collectors.toList());
                System.out.println("[SECURITY] Unsafe aliases detected: " + unsafe
                    + " — query will run unhinted.");
            }

            // Step 5: Ask AI engine to optimise join order.
            //   tables          → aliases  (Python builds hint with these identifiers)
            //   real_names      → realNames (Python fetches schema stats with these)
            //   join_conditions → edge list (Python graph-aware sampler)
            //   aliases_safe    → false means Python must NOT emit a hint
            Map<String, Object> pythonPayload = new HashMap<>();
            pythonPayload.put("tables",          aliases);
            pythonPayload.put("real_names",      realNames);
            pythonPayload.put("join_conditions", joinConditions);
            pythonPayload.put("original_sql",    sql);
            pythonPayload.put("aliases_safe",    aliasesAreSafe);

            String pythonAiUrl = "http://localhost:8000/optimize";
            long _oh0 = System.nanoTime();
            ResponseEntity<Map> aiResponse = restTemplate.postForEntity(pythonAiUrl, pythonPayload, Map.class);
            double optimizeHttpMs = (System.nanoTime() - _oh0) / 1e6;

            String       optimizedSql = (String)       aiResponse.getBody().get("optimized_query");
            List<String> chosenOrder  = (List<String>) aiResponse.getBody().get("chosen_order");

            // Extract query_id only — log_prob_old is stored server-side.
            Object  queryIdObj = aiResponse.getBody().get("query_id");
            Integer queryId    = (queryIdObj instanceof Number) ? ((Number) queryIdObj).intValue() : null;

            // Extract /optimize server timings (null-safe — absent on older server versions).
            Map<?, ?> optimizeTimings = null;
            Object otObj = aiResponse.getBody().get("server_timings_ms");
            if (otObj instanceof Map<?, ?>) {
                optimizeTimings = (Map<?, ?>) otObj;
            }

            // Step 6: Execute query under statement_timeout guard
            QueryTelemetryService.TelemetryResult metrics = queryTelemetryService.executeAndTrack(optimizedSql);

            System.out.println("Wait Time:    " + metrics.waitTimeMs  + " ms");
            System.out.println("Exec Time:    " + (metrics.execTimeMs == QueryTelemetryService.EXEC_TIME_MISSING
                                                    ? "TIMEOUT" : metrics.execTimeMs + " ms"));
            System.out.println("Total:        " + metrics.latencyMs   + " ms");
            System.out.println("Connections:  " + metrics.activeConnections);

            // Step 6b: Carbon emission estimate (third pillar).
            // Returns CarbonResult.unavailable() when the query was cancelled.
            CarbonResult carbon = carbonEstimator.estimate(metrics);
            System.out.printf("CO₂:          %.4f g  (energy: %.6f kWh)%n",
                              carbon.co2Grams, carbon.energyKwh);

            // Step 7: Send feedback — log_prob_old NOT forwarded (read from server-side store).
            // feedbackHttpMs and feedbackTimings declared outside the try so they are
            // accessible when building the breakdown map below.
            double  feedbackHttpMs = 0.0;
            Map<?, ?> feedbackTimings = null;
            try {
                Map<String, Object> feedbackPayload = new HashMap<>();
                feedbackPayload.put("tables",             aliases);
                feedbackPayload.put("real_names",         realNames);
                feedbackPayload.put("chosen_order",       chosenOrder);
                feedbackPayload.put("latency_ms",         metrics.latencyMs);
                feedbackPayload.put("exec_time_ms",       metrics.execTimeMs);
                feedbackPayload.put("wait_time_ms",       metrics.waitTimeMs);
                feedbackPayload.put("active_connections", metrics.activeConnections);
                if (queryId != null) feedbackPayload.put("query_id", queryId);

                String feedbackUrl = "http://localhost:8000/feedback";
                long _fh0 = System.nanoTime();
                ResponseEntity<Map> feedbackResponse = restTemplate.postForEntity(feedbackUrl, feedbackPayload, Map.class);
                feedbackHttpMs = (System.nanoTime() - _fh0) / 1e6;

                if (feedbackResponse.getBody() != null) {
                    Object ftObj = feedbackResponse.getBody().get("server_timings_ms");
                    if (ftObj instanceof Map<?, ?>) {
                        feedbackTimings = (Map<?, ?>) ftObj;
                    }
                }

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

            // Step 8: Build latency_breakdown_ms — all Java-side stages, server timings
            // from /optimize and /feedback, and derived fields.
            double totalMs = (System.nanoTime() - _totalStart) / 1e6;

            // LinkedHashMap preserves insertion order in JSON for readability.
            Map<String, Object> breakdown = new LinkedHashMap<>();

            // ── Java-side stages ──────────────────────────────────────────────
            breakdown.put("firewall_ms",           firewallMs);
            breakdown.put("extract_ms",            extractMs);
            breakdown.put("optimize_http_ms",      optimizeHttpMs);
            breakdown.put("active_conn_query_ms",  metrics.activeConnQueryMs);
            breakdown.put("pool_wait_ms",          (double) metrics.waitTimeMs);
            breakdown.put("blks_before_query_ms",  metrics.blksBeforeQueryMs);
            breakdown.put("set_timeout_ms",        metrics.setTimeoutMs);
            breakdown.put("explain_call_ms",       metrics.explainCallMs);
            // exec_time_ms_gap = how much execTimeMs overshoots explain_call_ms
            // (i.e. the cost of SET + blks_before that fall inside execStart's window).
            double execGap = (metrics.execTimeMs == QueryTelemetryService.EXEC_TIME_MISSING)
                ? -1.0
                : ((double) metrics.execTimeMs - metrics.explainCallMs);
            breakdown.put("exec_time_ms_gap",      execGap);
            breakdown.put("blks_after_query_ms",   metrics.blksAfterQueryMs);
            breakdown.put("reset_ms",              metrics.resetMs);
            breakdown.put("feedback_http_ms",      feedbackHttpMs);
            breakdown.put("total_ms",              totalMs);

            // ── /optimize server timings (prefixed "opt_") ────────────────────
            if (optimizeTimings != null) {
                for (Map.Entry<?, ?> e : optimizeTimings.entrySet()) {
                    breakdown.put("opt_" + e.getKey(), e.getValue());
                }
            }

            // ── /feedback server timings (prefixed "fb_") ─────────────────────
            if (feedbackTimings != null) {
                for (Map.Entry<?, ?> e : feedbackTimings.entrySet()) {
                    breakdown.put("fb_" + e.getKey(), e.getValue());
                }
            }

            // ── Derived fields ────────────────────────────────────────────────
            // unaccounted_ms = total - sum of all named stage timers.
            // Captures: JSON parsing, Spring overhead, response building, etc.
            // NOTE: explain_call_ms is used, NOT exec_time_ms, to avoid double-counting.
            double sumStages = firewallMs + extractMs + optimizeHttpMs
                + metrics.activeConnQueryMs + (double) metrics.waitTimeMs
                + metrics.blksBeforeQueryMs + metrics.setTimeoutMs + metrics.explainCallMs
                + metrics.blksAfterQueryMs + metrics.resetMs + feedbackHttpMs;
            breakdown.put("unaccounted_ms", totalMs - sumStages);

            // optimize_hop_overhead_ms = round-trip HTTP cost minus server processing time.
            if (optimizeTimings != null) {
                Object srvOptObj = optimizeTimings.get("optimize_total");
                if (srvOptObj instanceof Number) {
                    breakdown.put("optimize_hop_overhead_ms",
                        optimizeHttpMs - ((Number) srvOptObj).doubleValue());
                }
            }

            // feedback_hop_overhead_ms = same concept for /feedback.
            if (feedbackTimings != null) {
                Object srvFbObj = feedbackTimings.get("feedback_total");
                if (srvFbObj instanceof Number) {
                    breakdown.put("feedback_hop_overhead_ms",
                        feedbackHttpMs - ((Number) srvFbObj).doubleValue());
                }
            }

            // Step 9: Return enriched response (latency + carbon + breakdown fields)
            Map<String, Object> finalResponse = new HashMap<>(aiResponse.getBody());
            finalResponse.put("latency_ms",            metrics.latencyMs);
            finalResponse.put("exec_time_ms",          metrics.execTimeMs);
            finalResponse.put("wait_time_ms",          metrics.waitTimeMs);
            finalResponse.put("active_connections",    metrics.activeConnections);
            finalResponse.put("aliases_safe",          aliasesAreSafe);
            // Carbon pillar fields
            finalResponse.put("co2_grams",             carbon.co2Grams);
            finalResponse.put("energy_kwh",            carbon.energyKwh);
            finalResponse.put("cpu_energy_kwh",        carbon.cpuEnergyKwh);
            finalResponse.put("io_energy_kwh",         carbon.ioEnergyKwh);
            // Latency breakdown (all stages + server timings + derived)
            finalResponse.put("latency_breakdown_ms",  breakdown);

            return ResponseEntity.ok(finalResponse);

        } catch (Exception e) {
            e.printStackTrace();
            return ResponseEntity.badRequest().body("ERROR: " + e.getMessage());
        }
    }
}
