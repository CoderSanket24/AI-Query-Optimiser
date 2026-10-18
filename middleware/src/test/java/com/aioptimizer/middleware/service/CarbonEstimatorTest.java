package com.aioptimizer.middleware.service;

import com.aioptimizer.middleware.config.CarbonProperties;
import com.aioptimizer.middleware.model.CarbonResult;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit tests for CarbonEstimator.
 *
 * No Spring context needed — CarbonProperties is constructed directly.
 * All expected values are hand-derived from the formulae in CarbonEstimator's
 * Javadoc so they serve as living documentation for the paper.
 *
 * Constants used:
 *   TDP        = 150 W
 *   Intensity  = 708 g/kWh
 *   SSD power  = 2.0 W/GB
 */
class CarbonEstimatorTest {

    private static final double TDP_WATTS         = 150.0;
    private static final double INTENSITY_G_PER_KWH = 708.0;
    private static final double SSD_WATT_PER_GB   = 2.0;
    private static final double TOLERANCE         = 1e-9;

    private CarbonEstimator estimator;

    @BeforeEach
    void setup() {
        CarbonProperties props = new CarbonProperties();
        props.setTdpWatts(TDP_WATTS);
        props.setIntensityGPerKwh(INTENSITY_G_PER_KWH);
        props.setSsdWattPerGb(SSD_WATT_PER_GB);
        estimator = new CarbonEstimator(props);
    }

    // ── helper: build a fake TelemetryResult ──────────────────────────────────

    private static QueryTelemetryService.TelemetryResult fakeTelemetry(
            long execTimeMs, long blksReadBefore, long blksReadAfter) {
        return new QueryTelemetryService.TelemetryResult(
                execTimeMs, execTimeMs, 0, 1,
                blksReadBefore, blksReadAfter);
    }

    // ── test 1: cancelled query → unavailable() ───────────────────────────────

    @Test
    void cancelledQuery_returnsUnavailable() {
        QueryTelemetryService.TelemetryResult telemetry =
            new QueryTelemetryService.TelemetryResult(
                0, QueryTelemetryService.EXEC_TIME_MISSING, 0, 1, 0, 0);

        CarbonResult result = estimator.estimate(telemetry);

        assertEquals(QueryTelemetryService.EXEC_TIME_MISSING, result.execTimeMs,
            "Unavailable result must carry EXEC_TIME_MISSING sentinel");
        assertEquals(0.0, result.co2Grams,  TOLERANCE, "CO₂ must be 0 for unavailable result");
        assertEquals(0.0, result.energyKwh, TOLERANCE, "Energy must be 0 for unavailable result");
    }

    // ── test 2: CPU-only query (no I/O delta) ─────────────────────────────────
    //
    // exec_time = 1000 ms = 1 s
    // cpu_kwh   = 1 s × 150 W / 3_600_000 = 150 / 3_600_000 = 4.16667e-5 kWh
    // io_kwh    = 0
    // co2       = 4.16667e-5 × 708 = 0.02949917 g

    @Test
    void cpuOnlyQuery_noIo_correctCarbonEstimate() {
        long execMs = 1_000L;  // 1 second
        QueryTelemetryService.TelemetryResult telemetry = fakeTelemetry(execMs, 100L, 100L);

        CarbonResult result = estimator.estimate(telemetry);

        double expectedCpuKwh = (execMs / 1_000.0) * TDP_WATTS / 3_600_000.0;
        double expectedCo2    = expectedCpuKwh * INTENSITY_G_PER_KWH;

        assertEquals(expectedCpuKwh, result.cpuEnergyKwh, TOLERANCE, "CPU energy mismatch");
        assertEquals(0.0,            result.ioEnergyKwh,  TOLERANCE, "I/O energy must be 0");
        assertEquals(expectedCpuKwh, result.energyKwh,    TOLERANCE, "Total energy mismatch");
        assertEquals(expectedCo2,    result.co2Grams,     TOLERANCE, "CO₂ mismatch");
        assertTrue(result.co2Grams > 0, "CO₂ must be positive for a real execution");
    }

    // ── test 3: query with I/O reads ──────────────────────────────────────────
    //
    // exec_time  = 500 ms = 0.5 s
    // blks_delta = 256 blocks = 256 × 4096 bytes = 1_048_576 bytes = ~1.049e-3 GB
    // cpu_kwh    = 0.5 × 150 / 3_600_000 = 2.08333e-5 kWh
    // io_kwh     = (1_048_576 / 1e9) × 2.0 / 3_600_000 = 5.82533e-13 kWh
    // co2        = (cpu_kwh + io_kwh) × 708

    @Test
    void queryWithIo_correctCarbonEstimate() {
        long execMs     = 500L;
        long blksBefore = 1_000L;
        long blksAfter  = 1_256L;   // delta = 256 blocks
        QueryTelemetryService.TelemetryResult telemetry = fakeTelemetry(execMs, blksBefore, blksAfter);

        CarbonResult result = estimator.estimate(telemetry);

        long   blksDelta     = 256L;
        double cpuKwh        = (execMs / 1_000.0) * TDP_WATTS / 3_600_000.0;
        double ioKwh         = (blksDelta * 4096.0 / 1_000_000_000.0) * SSD_WATT_PER_GB / 3_600_000.0;
        double totalKwh      = cpuKwh + ioKwh;
        double co2            = totalKwh * INTENSITY_G_PER_KWH;

        assertEquals(blksDelta, result.blksReadDelta, "Block read delta mismatch");
        assertEquals(cpuKwh,    result.cpuEnergyKwh,  TOLERANCE, "CPU energy mismatch");
        assertEquals(ioKwh,     result.ioEnergyKwh,   TOLERANCE, "I/O energy mismatch");
        assertEquals(totalKwh,  result.energyKwh,     TOLERANCE, "Total energy mismatch");
        assertEquals(co2,       result.co2Grams,      TOLERANCE, "CO₂ mismatch");
    }

    // ── test 4: negative blks_read delta is clamped to 0 ─────────────────────
    // Can happen when another session causes a very large read between snapshots.

    @Test
    void negativeBlksDelta_clampedToZero_noCpuUnaffected() {
        long execMs = 200L;
        QueryTelemetryService.TelemetryResult telemetry = fakeTelemetry(execMs, 500L, 300L);  // after < before

        CarbonResult result = estimator.estimate(telemetry);

        assertEquals(0L,  result.blksReadDelta, "Negative delta must be clamped to 0");
        assertEquals(0.0, result.ioEnergyKwh,   TOLERANCE, "I/O energy must be 0 when delta clamped");
        // CPU energy should still be computed correctly
        double expectedCpuKwh = (execMs / 1_000.0) * TDP_WATTS / 3_600_000.0;
        assertEquals(expectedCpuKwh, result.cpuEnergyKwh, TOLERANCE, "CPU energy must not be affected");
    }

    // ── test 5: session totals accumulate across calls ────────────────────────

    @Test
    void sessionTotals_accumulateCorrectly() {
        QueryTelemetryService.TelemetryResult t1 = fakeTelemetry(1_000L, 0L, 0L);
        QueryTelemetryService.TelemetryResult t2 = fakeTelemetry(2_000L, 0L, 0L);

        CarbonResult r1 = estimator.estimate(t1);
        CarbonResult r2 = estimator.estimate(t2);

        double expectedTotal = r1.co2Grams + r2.co2Grams;

        assertEquals(2, estimator.getTotalQueries(), "Should count 2 queries");
        assertEquals(expectedTotal, estimator.getTotalCo2Grams(), TOLERANCE, "Cumulative CO₂ mismatch");
        assertEquals(expectedTotal / 2.0, estimator.getAvgCo2PerQuery(), TOLERANCE, "Average CO₂ mismatch");
    }

    // ── test 6: select 1 produces tiny but positive CO₂ ─────────────────────
    // Sanity check: a trivial 1 ms query should produce < 0.01 g

    @Test
    void trivialQuery_producesSmallButPositiveCo2() {
        QueryTelemetryService.TelemetryResult telemetry = fakeTelemetry(1L, 0L, 0L);

        CarbonResult result = estimator.estimate(telemetry);

        assertTrue(result.co2Grams > 0,    "CO₂ must be positive even for 1 ms queries");
        assertTrue(result.co2Grams < 0.01, "CO₂ for a 1 ms query must be < 0.01 g");
    }
}
