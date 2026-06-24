"""
simulator.py — pure simulation engine for road-tunnel electrical loads.

No Streamlit dependency. Importable as a regular Python module.
Vectorized NumPy/pandas. Suitable for batch Monte Carlo studies.

Project
-------
Part of the **SmartTunnel** research project, a collaboration between:

  * Haytham El-Houari, PhD — Université Sidi Mohamed Ben Abdellah (USMBA),
    Fès, Morocco.
  * Cyril Voyant, Professor — Mines Paris – PSL, OIE Laboratory (Centre
    Observation, Impacts, Énergie), France.

License: MIT. See CITATION.cff for the canonical citation entry.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


# =============================================================================
# Engineering constants — calibrate against measured data for real studies
# =============================================================================

ALTITUDE_REF_M = 500.0
ALTITUDE_DENOM = 6000.0
DEPTH_CAP_M = 800.0
DEPTH_DENOM = 6000.0
SLOPE_DENOM = 20.0

SEASON_NAMES = np.array(["winter", "spring", "summer", "autumn"])
SEASON_FACTORS = np.array([1.12, 0.98, 0.92, 1.03])

DAYLIGHT_PARAMS = {
    "winter": (8.0, 17.0),
    "spring": (6.5, 20.0),
    "summer": (5.5, 21.5),
    "autumn": (7.0, 18.5),
}
DAYLIGHT_SHARPNESS = 3.0

# (kW/km/lane, traffic_coupling, min_fraction)
LIGHTING_PARAMS = {
    "LED adaptive": (16.0, 0.18, 0.20),
    "LED fixed":    (22.0, 0.06, 0.28),
    "mixed":        (28.0, 0.04, 0.32),
    "sodium fixed": (35.0, 0.03, 0.40),
}

VENTILATION_PARAMS = {
    "natural/low ventilation": 45.0,
    "longitudinal":            120.0,
    "semi-transverse":         200.0,
    "transverse":              320.0,
}

CONTEXT_MULT = {
    "urban":      1.20,
    "peri-urban": 1.00,
    "rural":      0.78,
}

VENT_MIN_FRACTION = 0.15
DAILY_TRAFFIC_SIGMA = 0.06
AUX_NOISE_SIGMA = 0.025


# =============================================================================
# Vectorized helpers
# =============================================================================

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def gaussian_peak(hour: np.ndarray, center: float, width: float, amplitude: float) -> np.ndarray:
    return amplitude * np.exp(-0.5 * ((hour - center) / width) ** 2)


def assign_season_idx(month: np.ndarray) -> np.ndarray:
    return np.where(
        (month == 12) | (month <= 2), 0,
        np.where(month <= 5, 1,
                 np.where(month <= 8, 2, 3))
    )


def daylight_array(hour: np.ndarray, season_idx: np.ndarray) -> np.ndarray:
    sunrises = np.array([DAYLIGHT_PARAMS[s][0] for s in SEASON_NAMES])
    sunsets = np.array([DAYLIGHT_PARAMS[s][1] for s in SEASON_NAMES])
    sr = sunrises[season_idx]
    ss = sunsets[season_idx]
    return sigmoid((hour - sr) * DAYLIGHT_SHARPNESS) * sigmoid((ss - hour) * DAYLIGHT_SHARPNESS)


def weekday_factor_array(dow: np.ndarray) -> np.ndarray:
    return np.where(dow < 5, 1.00, np.where(dow == 5, 0.86, 0.78))


# =============================================================================
# Configuration
# =============================================================================

@dataclass(frozen=True)
class TunnelConfig:
    # Geometry
    length_m: float
    n_tubes: int
    n_lanes_per_tube: int
    altitude_m: float
    max_depth_m: float
    gradient_percent: float
    # Operating context
    tunnel_context: str
    lighting_type: str
    ventilation_type: str
    aux_kw_per_km_tube: float
    base_fixed_kw: float
    # Traffic
    traffic_level: float
    morning_peak_hour: float
    evening_peak_hour: float
    peak_width_h: float
    traffic_sensitivity: float
    # Stochastic
    noise_sigma: float
    pollution_probability_per_day: float
    accident_probability_per_day: float
    pollution_sensitivity: float
    accident_sensitivity: float

    def __post_init__(self):
        if self.length_m <= 0:
            raise ValueError("length_m must be > 0")
        if not (1 <= self.n_tubes <= 8):
            raise ValueError("n_tubes out of range")
        if not (1 <= self.n_lanes_per_tube <= 6):
            raise ValueError("n_lanes_per_tube out of range")
        if not (0 <= self.pollution_probability_per_day <= 1):
            raise ValueError("pollution_probability_per_day must be in [0,1]")
        if not (0 <= self.accident_probability_per_day <= 1):
            raise ValueError("accident_probability_per_day must be in [0,1]")
        if self.noise_sigma < 0:
            raise ValueError("noise_sigma must be >= 0")
        if self.tunnel_context not in CONTEXT_MULT:
            raise ValueError(f"unknown tunnel_context: {self.tunnel_context}")
        if self.lighting_type not in LIGHTING_PARAMS:
            raise ValueError(f"unknown lighting_type: {self.lighting_type}")
        if self.ventilation_type not in VENTILATION_PARAMS:
            raise ValueError(f"unknown ventilation_type: {self.ventilation_type}")


def reference_traffic_norm(cfg: TunnelConfig) -> float:
    """
    Compute normalization constant from a 1-day deterministic reference.
    This makes traffic_level a true scaling parameter and decouples
    normalization from simulation length.
    """
    h = np.linspace(0, 24, 144, endpoint=False)
    ref = (
        0.16 + 0.06 * np.cos(2 * np.pi * (h - 3.0) / 24.0)
        + gaussian_peak(h, cfg.morning_peak_hour, cfg.peak_width_h, 1.00)
        + gaussian_peak(h, cfg.evening_peak_hour, cfg.peak_width_h * 1.15, 1.10)
    )
    if cfg.tunnel_context == "urban":
        ref = ref + gaussian_peak(h, 18.7, 2.5, 0.25)
    return float(max(np.quantile(ref, 0.98), 1e-6))


# =============================================================================
# Event sampling — vectorized with cross-midnight spillover
# =============================================================================

def _sample_event_flags(
    rng: np.random.Generator,
    hour: np.ndarray,
    day_id: np.ndarray,
    n_days: int,
    prob: float,
    start_low: float,
    start_high: float,
    dur_low: float,
    dur_high: float,
) -> np.ndarray:
    """Per-step event activation mask. O(n_steps), no Python day loop."""
    if prob <= 0 or n_days == 0:
        return np.zeros(hour.size, dtype=bool)

    occurs = rng.random(n_days) < prob
    starts = rng.uniform(start_low, start_high, n_days)
    durs = rng.uniform(dur_low, dur_high, n_days)

    occurs_step = occurs[day_id]
    start_step = starts[day_id]
    dur_step = durs[day_id]
    same_day = occurs_step & (hour >= start_step) & (hour < np.minimum(start_step + dur_step, 24.0))

    prev = day_id - 1
    prev_valid = prev >= 0
    prev_clip = np.maximum(prev, 0)
    prev_occurs = occurs[prev_clip]
    prev_start = starts[prev_clip]
    prev_dur = durs[prev_clip]
    overflow = prev_start + prev_dur - 24.0
    spillover = prev_valid & prev_occurs & (overflow > 0) & (hour < overflow)

    return same_day | spillover


# =============================================================================
# Single realization
# =============================================================================

def simulate_one_realization(
    start_date: pd.Timestamp,
    n_days: int,
    freq_minutes: int,
    cfg: TunnelConfig,
    seed: int,
) -> pd.DataFrame:
    """Run one Monte Carlo realization, return a tidy DataFrame."""
    rng = np.random.default_rng(seed)

    n_steps_per_day = (24 * 60) // freq_minutes
    n_points = n_days * n_steps_per_day
    idx = pd.date_range(start=start_date, periods=n_points, freq=f"{freq_minutes}min")

    hour = idx.hour.values + idx.minute.values / 60.0
    dow = idx.dayofweek.values
    month = idx.month.values
    season_idx = assign_season_idx(month)
    day_id = np.repeat(np.arange(n_days), n_steps_per_day)

    length_km = cfg.length_m / 1000.0
    total_lanes = cfg.n_tubes * cfg.n_lanes_per_tube
    altitude_factor = 1.0 + max(cfg.altitude_m - ALTITUDE_REF_M, 0.0) / ALTITUDE_DENOM
    depth_factor = 1.0 + min(cfg.max_depth_m, DEPTH_CAP_M) / DEPTH_DENOM
    slope_factor = 1.0 + abs(cfg.gradient_percent) / SLOPE_DENOM

    daily_mult = rng.normal(1.0, DAILY_TRAFFIC_SIGMA, n_days)
    daily_mult_per_step = daily_mult[day_id]

    morning = gaussian_peak(hour, cfg.morning_peak_hour, cfg.peak_width_h, 1.00)
    evening = gaussian_peak(hour, cfg.evening_peak_hour, cfg.peak_width_h * 1.15, 1.10)
    night_floor = 0.16 + 0.06 * np.cos(2 * np.pi * (hour - 3.0) / 24.0)
    raw_traffic = night_floor + morning + evening
    if cfg.tunnel_context == "urban":
        raw_traffic = raw_traffic + gaussian_peak(hour, 18.7, 2.5, 0.25)

    norm_constant = reference_traffic_norm(cfg)
    context_mult = CONTEXT_MULT[cfg.tunnel_context]
    season_mult = SEASON_FACTORS[season_idx]
    day_mult = weekday_factor_array(dow)

    traffic = (raw_traffic / norm_constant) * context_mult * season_mult * day_mult \
        * cfg.traffic_level * daily_mult_per_step
    traffic = np.clip(traffic, 0.0, 2.0)

    daylight = daylight_array(hour, season_idx)

    spec_kw, traffic_coupling, min_frac = LIGHTING_PARAMS[cfg.lighting_type]
    lighting_installed_kw = spec_kw * length_km * total_lanes
    lighting_factor = (
        min_frac
        + (1.0 - min_frac) * (1.0 - daylight)
        + traffic_coupling * traffic
    )
    lighting_factor = np.minimum(lighting_factor, 1.0)
    lighting_kw = lighting_installed_kw * lighting_factor

    ventilation_scale = VENTILATION_PARAMS[cfg.ventilation_type] * length_km * cfg.n_tubes
    ventilation_kw = (
        ventilation_scale
        * (VENT_MIN_FRACTION + cfg.traffic_sensitivity * traffic)
        * altitude_factor
        * slope_factor
    )

    pollution_event = _sample_event_flags(
        rng, hour, day_id, n_days,
        prob=cfg.pollution_probability_per_day,
        start_low=7.0, start_high=18.0,
        dur_low=2.0, dur_high=8.0,
    )
    accident_event = _sample_event_flags(
        rng, hour, day_id, n_days,
        prob=cfg.accident_probability_per_day,
        start_low=6.5, start_high=20.0,
        dur_low=0.5, dur_high=3.0,
    )

    ventilation_kw = ventilation_kw * (
        1.0
        + cfg.pollution_sensitivity * pollution_event.astype(float)
        + cfg.accident_sensitivity * accident_event.astype(float)
    )

    aux_base = cfg.aux_kw_per_km_tube * length_km * cfg.n_tubes * depth_factor
    auxiliary_kw = aux_base * rng.normal(1.0, AUX_NOISE_SIGMA, n_points)

    deterministic_kw = cfg.base_fixed_kw + lighting_kw + ventilation_kw + auxiliary_kw
    multiplicative_noise = rng.normal(0.0, cfg.noise_sigma, n_points)
    power_kw = np.maximum(deterministic_kw * (1.0 + multiplicative_noise), 0.0)

    return pd.DataFrame({
        "timestamp": idx,
        "season": SEASON_NAMES[season_idx],
        "hour_int": idx.hour.values,
        "dayofweek": dow,
        "day_type": np.where(dow < 5, "weekday", "weekend"),
        "traffic_index": traffic.astype(np.float32),
        "pollution_event": pollution_event.astype(np.int8),
        "accident_event": accident_event.astype(np.int8),
        "lighting_kw": lighting_kw.astype(np.float32),
        "ventilation_kw": ventilation_kw.astype(np.float32),
        "auxiliary_kw": auxiliary_kw.astype(np.float32),
        "power_kw": power_kw.astype(np.float32),
        "energy_kwh": (power_kw * freq_minutes / 60.0).astype(np.float32),
    })


# =============================================================================
# Monte Carlo wrapper
# =============================================================================

def run_monte_carlo(
    cfg: TunnelConfig,
    start_date: pd.Timestamp,
    n_days: int,
    freq_minutes: int,
    n_runs: int,
    base_seed: int,
) -> dict:
    """
    Run N Monte Carlo realizations.

    Returns a dict containing:
      - 'representative' : full DataFrame of run #0 (decomposition view)
      - 'kpis'           : DataFrame, one row per run
      - 'envelope'       : DataFrame with median / p10 / p90 / mean of total
                           power, hourly resampled across all MC runs
      - 'season_profiles': long-format DataFrame, mean hourly profile per
                           (season, hour, run)
    """
    kpis_rows = []
    seasonal_profiles_list = []
    power_traces_hourly = []
    representative_df = None
    hourly_index_ref = None

    for i in range(n_runs):
        seed = base_seed + i
        df = simulate_one_realization(start_date, n_days, freq_minutes, cfg, seed)

        if i == 0:
            representative_df = df

        total_mwh = float(df["energy_kwh"].sum() / 1000.0)
        annualized_mwh = total_mwh * 365.0 / n_days
        peak_kw = float(df["power_kw"].max())
        mean_kw = float(df["power_kw"].mean())
        load_factor = mean_kw / peak_kw if peak_kw > 0 else float("nan")
        specific_kwh_m_year = annualized_mwh * 1000.0 / cfg.length_m

        kpis_rows.append({
            "run": i,
            "seed": seed,
            "total_mwh": total_mwh,
            "annualized_mwh": annualized_mwh,
            "peak_kw": peak_kw,
            "mean_kw": mean_kw,
            "load_factor": load_factor,
            "specific_kwh_m_year": specific_kwh_m_year,
            "n_pollution_events": int((df["pollution_event"].diff() == 1).sum()),
            "n_accident_events": int((df["accident_event"].diff() == 1).sum()),
        })

        df_hourly = df.set_index("timestamp")["power_kw"].resample("1h").mean()
        power_traces_hourly.append(df_hourly.values.astype(np.float32))
        if hourly_index_ref is None:
            hourly_index_ref = df_hourly.index

        sp = df.groupby(["season", "hour_int"], observed=True)["power_kw"].mean().reset_index()
        sp["run"] = i
        seasonal_profiles_list.append(sp)

    kpis_df = pd.DataFrame(kpis_rows)
    traces_arr = np.array(power_traces_hourly)
    envelope = pd.DataFrame({
        "timestamp": hourly_index_ref,
        "median": np.median(traces_arr, axis=0),
        "p10":    np.percentile(traces_arr, 10, axis=0),
        "p90":    np.percentile(traces_arr, 90, axis=0),
        "mean":   np.mean(traces_arr, axis=0),
    })
    season_profiles_long = pd.concat(seasonal_profiles_list, ignore_index=True)

    return {
        "representative": representative_df,
        "kpis": kpis_df,
        "envelope": envelope,
        "season_profiles": season_profiles_long,
    }
