package com.aioptimizer.middleware.config;

import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.stereotype.Component;

/**
 * Carbon emission tracking configuration.
 *
 * Values are read from application.properties (prefix "carbon") so they can
 * be overridden per deployment environment without recompiling.
 *
 * Defaults (set here as a safety net — application.properties takes precedence):
 *   tdpWatts            = 150    W   typical mid-range server CPU
 *   intensityGPerKwh    = 708.0  g/kWh  India grid average (CEA 2023)
 *   ssdWattPerGb        = 2.0    W/GB   SSD sequential read power
 *
 * For a research paper, cite the intensity value and note it is configurable.
 */
@Component
@ConfigurationProperties(prefix = "carbon")
public class CarbonProperties {

    /** Server CPU thermal design power in watts. */
    private double tdpWatts = 150.0;

    /** Grid carbon intensity in grams CO₂eq per kWh. */
    private double intensityGPerKwh = 708.0;

    /** SSD I/O power in watts per GB read. */
    private double ssdWattPerGb = 2.0;

    // ── Getters and setters (required by @ConfigurationProperties) ────────────

    public double getTdpWatts()           { return tdpWatts; }
    public void   setTdpWatts(double v)   { this.tdpWatts = v; }

    public double getIntensityGPerKwh()          { return intensityGPerKwh; }
    public void   setIntensityGPerKwh(double v)  { this.intensityGPerKwh = v; }

    public double getSsdWattPerGb()          { return ssdWattPerGb; }
    public void   setSsdWattPerGb(double v)  { this.ssdWattPerGb = v; }
}
