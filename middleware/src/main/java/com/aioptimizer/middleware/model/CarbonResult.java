package com.aioptimizer.middleware.model;

/**
 * Per-query carbon emission estimate produced by CarbonEstimator.
 *
 * Fields:
 *   energyKwh    -- total energy consumed by this query (CPU + I/O), in kilowatt-hours
 *   co2Grams     -- CO₂ equivalent in grams  (energyKwh × grid carbon intensity)
 *   cpuEnergyKwh -- CPU component only        (exec_time × TDP)
 *   ioEnergyKwh  -- I/O component only        (blks_read delta × SSD power model)
 *   blksReadDelta -- raw buffer reads observed during query execution
 *   execTimeMs   -- execution time used for CPU energy calculation (mirrors TelemetryResult)
 */
public class CarbonResult {

    public final double energyKwh;
    public final double co2Grams;
    public final double cpuEnergyKwh;
    public final double ioEnergyKwh;
    public final long   blksReadDelta;
    public final long   execTimeMs;

    public CarbonResult(double energyKwh,
                        double co2Grams,
                        double cpuEnergyKwh,
                        double ioEnergyKwh,
                        long   blksReadDelta,
                        long   execTimeMs) {
        this.energyKwh    = energyKwh;
        this.co2Grams     = co2Grams;
        this.cpuEnergyKwh = cpuEnergyKwh;
        this.ioEnergyKwh  = ioEnergyKwh;
        this.blksReadDelta = blksReadDelta;
        this.execTimeMs   = execTimeMs;
    }

    /** Sentinel: returned when exec_time is missing (query was cancelled). */
    public static CarbonResult unavailable() {
        return new CarbonResult(0, 0, 0, 0, 0,
                                com.aioptimizer.middleware.service.QueryTelemetryService.EXEC_TIME_MISSING);
    }
}
