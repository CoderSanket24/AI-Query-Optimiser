package com.aioptimizer.middleware.controller;

import com.aioptimizer.middleware.service.CarbonEstimator;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import java.util.LinkedHashMap;
import java.util.Map;

/**
 * CarbonController — REST surface for the Carbon Emission Tracking pillar.
 *
 * Endpoints:
 *   GET /api/carbon/summary
 *     Returns session-wide totals since the last application restart.
 *     Response fields:
 *       total_co2_grams      — cumulative CO₂ for all tracked queries (g)
 *       total_energy_kwh     — cumulative energy consumed (kWh)
 *       total_queries        — number of completed (non-cancelled) queries tracked
 *       avg_co2_per_query_g  — mean CO₂ per query (g); 0 when no queries yet
 */
@RestController
@RequestMapping("api/carbon")
public class CarbonController {

    private final CarbonEstimator carbonEstimator;

    public CarbonController(CarbonEstimator carbonEstimator) {
        this.carbonEstimator = carbonEstimator;
    }

    @GetMapping("/summary")
    public ResponseEntity<Map<String, Object>> summary() {
        Map<String, Object> body = new LinkedHashMap<>();
        body.put("total_co2_grams",     carbonEstimator.getTotalCo2Grams());
        body.put("total_energy_kwh",    carbonEstimator.getTotalEnergyKwh());
        body.put("total_queries",       carbonEstimator.getTotalQueries());
        body.put("avg_co2_per_query_g", carbonEstimator.getAvgCo2PerQuery());
        return ResponseEntity.ok(body);
    }
}
