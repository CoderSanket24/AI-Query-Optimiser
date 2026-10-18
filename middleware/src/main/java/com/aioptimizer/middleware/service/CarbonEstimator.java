package com.aioptimizer.middleware.service;

import com.aioptimizer.middleware.config.CarbonProperties;
import com.aioptimizer.middleware.model.CarbonResult;
import org.springframework.stereotype.Service;

import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.DoubleAdder;

/**
 * CarbonEstimator — Third Pillar: Carbon Emission Tracking
 * ─────────────────────────────────────────────────────────
 * Converts raw telemetry (exec time + I/O buffer reads) into an energy and CO₂
 * estimate for every executed query.
 *
 * Model (transparent for the research paper):
 *
 *   CPU energy:
 *     cpu_kwh = (exec_time_ms / 1000.0) × TDP_WATTS / 3_600_000.0
 *     (TDP_WATTS is the server CPU's thermal design power; exec_time is the
 *      actual wall-clock time PostgreSQL spent on the query)
 *
 *   I/O energy:
 *     io_kwh  = (blks_read_delta × 4096 bytes / 1e9 GB) × (SSD_WATT_PER_GB / 3_600_000.0)
 *     (blks_read_delta comes from pg_stat_database — cumulative physical disk reads
 *      attributed to this database during the query's execution window)
 *
 *   Total:
 *     total_kwh = cpu_kwh + io_kwh
 *     co2_grams = total_kwh × INTENSITY_G_PER_KWH
 *
 * Limitations (note in the paper):
 *   - blks_read is DB-wide, not per-query, so concurrent queries add noise.
 *   - TDP is a peak rating; actual draw is typically 40–70% of TDP at moderate load.
 *   - The model is suitable for relative comparisons (plan A vs. plan B), not
 *     absolute measurement.
 *
 * Session totals (for /api/carbon/summary):
 *   - Tracked via lock-free atomics; thread-safe for concurrent requests.
 */
@Service
public class CarbonEstimator {

    private final CarbonProperties props;

    // ── Session-wide accumulators (reset on restart) ───────────────────────
    private final DoubleAdder totalCo2Grams  = new DoubleAdder();
    private final DoubleAdder totalEnergyKwh = new DoubleAdder();
    private final AtomicLong  totalQueries   = new AtomicLong(0);

    public CarbonEstimator(CarbonProperties props) {
        this.props = props;
    }

    /**
     * Compute the carbon footprint of one query execution.
     *
     * Returns {@link CarbonResult#unavailable()} when exec_time_ms is the
     * EXEC_TIME_MISSING sentinel (cancelled query) — carbon cannot be estimated
     * without an execution time.
     */
    public CarbonResult estimate(QueryTelemetryService.TelemetryResult telemetry) {
        if (telemetry.execTimeMs == QueryTelemetryService.EXEC_TIME_MISSING) {
            return CarbonResult.unavailable();
        }

        long blksReadDelta = Math.max(0L, telemetry.blksReadAfter - telemetry.blksReadBefore);

        // CPU energy: exec time × TDP fraction
        // 3_600_000 converts (W × s) → kWh:  1 kWh = 3,600,000 J = 3,600,000 W·s
        double cpuEnergyKwh = (telemetry.execTimeMs / 1_000.0)
                              * props.getTdpWatts()
                              / 3_600_000.0;

        // I/O energy: physical blocks read × 4 KB each → GB → kWh
        // 1 block = 4096 bytes = 4096 / 1e9 GB
        // Energy = GB × (W/GB) / 3_600_000 (W·s → kWh)
        double ioEnergyKwh  = (blksReadDelta * 4096.0 / 1_000_000_000.0)
                              * props.getSsdWattPerGb()
                              / 3_600_000.0;

        double totalEnergyKwh = cpuEnergyKwh + ioEnergyKwh;
        double co2Grams       = totalEnergyKwh * props.getIntensityGPerKwh();

        // Accumulate session totals
        this.totalCo2Grams.add(co2Grams);
        this.totalEnergyKwh.add(totalEnergyKwh);
        this.totalQueries.incrementAndGet();

        System.out.printf("[Carbon] cpu=%.6f kWh  io=%.6f kWh  total=%.6f kWh  co2=%.4f g%n",
                          cpuEnergyKwh, ioEnergyKwh, totalEnergyKwh, co2Grams);

        return new CarbonResult(totalEnergyKwh, co2Grams,
                                cpuEnergyKwh, ioEnergyKwh,
                                blksReadDelta, telemetry.execTimeMs);
    }

    // ── Session summary accessors (used by CarbonController) ──────────────

    public double getTotalCo2Grams()  { return totalCo2Grams.sum(); }
    public double getTotalEnergyKwh() { return totalEnergyKwh.sum(); }
    public long   getTotalQueries()   { return totalQueries.get(); }

    public double getAvgCo2PerQuery() {
        long n = totalQueries.get();
        return (n == 0) ? 0.0 : totalCo2Grams.sum() / n;
    }
}
