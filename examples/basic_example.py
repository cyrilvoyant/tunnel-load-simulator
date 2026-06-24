"""
Basic example for Tunnel Load Simulator.
Runs a single realization and prints key indicators.
"""
import pandas as pd
from tunnel_load_simulator.simulator import TunnelConfig, simulate_one_realization

cfg = TunnelConfig(
    length_m=1500,
    n_tubes=2,
    n_lanes_per_tube=2,
    altitude_m=300,
    max_depth_m=80,
    gradient_percent=2,
    tunnel_context="peri-urban",
    lighting_type="LED adaptive",
    ventilation_type="longitudinal",
    aux_kw_per_km_tube=35,
    base_fixed_kw=40,
    traffic_level=1.0,
    morning_peak_hour=8,
    evening_peak_hour=18,
    peak_width_h=1.4,
    traffic_sensitivity=0.65,
    noise_sigma=0.06,
    pollution_probability_per_day=0.05,
    accident_probability_per_day=0.015,
    pollution_sensitivity=0.55,
    accident_sensitivity=0.75,
)

df = simulate_one_realization(
    pd.Timestamp("2024-01-01"),
    365,
    10,
    cfg,
    seed=42,
)

annual_energy_mwh = df["energy_kwh"].sum() / 1000
peak_power_kw = df["power_kw"].max()
mean_power_kw = df["power_kw"].mean()
load_factor = mean_power_kw / peak_power_kw

print("=== Single realization — Key indicators ===")
print(f"  Annual energy : {annual_energy_mwh:.1f} MWh")
print(f"  Peak power    : {peak_power_kw:.1f} kW")
print(f"  Mean power    : {mean_power_kw:.1f} kW")
print(f"  Load factor   : {load_factor:.3f}")
