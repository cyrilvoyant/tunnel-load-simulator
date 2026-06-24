"""Validation tests for the simulator engine (no Streamlit needed)."""

import time
import numpy as np
import pandas as pd

from tunnel_load_simulator.simulator import (
    LIGHTING_PARAMS,
    TunnelConfig,
    run_monte_carlo,
    simulate_one_realization,
)


def make_default_cfg(**overrides) -> TunnelConfig:
    base = dict(
        length_m=1500.0, n_tubes=2, n_lanes_per_tube=2,
        altitude_m=300.0, max_depth_m=80.0, gradient_percent=2.0,
        tunnel_context="peri-urban", lighting_type="LED adaptive",
        ventilation_type="longitudinal",
        aux_kw_per_km_tube=35.0, base_fixed_kw=40.0,
        traffic_level=1.0, morning_peak_hour=8.0, evening_peak_hour=18.0,
        peak_width_h=1.4, traffic_sensitivity=0.65,
        noise_sigma=0.06,
        pollution_probability_per_day=0.05, accident_probability_per_day=0.015,
        pollution_sensitivity=0.55, accident_sensitivity=0.75,
    )
    base.update(overrides)
    return TunnelConfig(**base)


print("=" * 70)
print("Test 1 — Single realization, 1 year, 10-min step")
print("=" * 70)
cfg = make_default_cfg()
t0 = time.time()
df = simulate_one_realization(pd.Timestamp("2024-01-01"), 365, 10, cfg, seed=42)
dt = (time.time() - t0) * 1000
print(f"  {len(df):,} rows in {dt:.0f} ms")
print(f"  Power kW: min={df['power_kw'].min():.1f} mean={df['power_kw'].mean():.1f} "
      f"max={df['power_kw'].max():.1f}")
print(f"  Total energy: {df['energy_kwh'].sum()/1000:.1f} MWh")
print(f"  Pollution-active steps: {int((df['pollution_event']==1).sum()):,}")
print(f"  Accident-active steps:  {int((df['accident_event']==1).sum()):,}")
assert df["power_kw"].min() >= 0
assert len(df) == 365 * 144

print()
print("=" * 70)
print("Test 2 — Performance: 2 years 10-min step")
print("=" * 70)
t0 = time.time()
df2 = simulate_one_realization(pd.Timestamp("2024-01-01"), 730, 10, cfg, seed=42)
dt = (time.time() - t0) * 1000
print(f"  {len(df2):,} rows in {dt:.0f} ms")

print()
print("=" * 70)
print("Test 3 — Seasonal profiles ARE different (the bug fix!)")
print("=" * 70)
sp = df.groupby(["season", "hour_int"])["power_kw"].mean().unstack(level=0)
for s in ["winter", "spring", "summer", "autumn"]:
    if s in sp.columns:
        col = sp[s]
        print(f"  {s:8s}: hour 00 = {col.iloc[0]:6.1f} kW  "
              f"hour 12 = {col.iloc[12]:6.1f} kW  hour 18 = {col.iloc[18]:6.1f} kW")
cross_std = sp.std(axis=1).mean()
print(f"  Cross-season std (mean over hours): {cross_std:.2f} kW  "
      f"-> {'DIFFERENT ✓' if cross_std > 1 else 'IDENTICAL ✗'}")
assert cross_std > 1, "Seasons should differ in mean power"

print()
print("=" * 70)
print("Test 4 — traffic_level effectively scales traffic (was bugged before)")
print("=" * 70)
levels = []
for tl in [0.5, 1.0, 1.5]:
    cfg_tl = make_default_cfg(traffic_level=tl)
    df_tl = simulate_one_realization(pd.Timestamp("2024-01-01"), 30, 10, cfg_tl, seed=42)
    levels.append((tl, df_tl["traffic_index"].mean(), df_tl["power_kw"].mean()))
    print(f"  traffic_level={tl}: mean traffic_index={df_tl['traffic_index'].mean():.3f}  "
          f"mean power={df_tl['power_kw'].mean():.1f} kW")
# 1.5 should give noticeably more than 0.5
assert levels[2][1] > levels[0][1] * 1.5, "traffic_level should scale traffic_index roughly linearly"

print()
print("=" * 70)
print("Test 5 — Cross-midnight event spillover")
print("=" * 70)
# Force pollution every day; events can start as late as 18h with 8h duration -> spill to next day
# Theoretical spillover probability per event: ~3% (need ≥100 days for reliable sample)
cfg_ev = make_default_cfg(pollution_probability_per_day=1.0)
df_ev = simulate_one_realization(pd.Timestamp("2024-01-01"), 100, 10, cfg_ev, seed=1)
early = df_ev[(df_ev["hour_int"] < 3) & (df_ev["pollution_event"] == 1)]
print(f"  Pollution-active steps in [00:00–03:00] (spillover) over 100 days: {len(early)}")
print(f"  Total pollution-active steps: {int((df_ev['pollution_event']==1).sum()):,}")
assert len(early) > 0, "Some events should spill past midnight"

print()
print("=" * 70)
print("Test 6 — Seed reproducibility")
print("=" * 70)
df_a = simulate_one_realization(pd.Timestamp("2024-01-01"), 30, 10, cfg, seed=1)
df_b = simulate_one_realization(pd.Timestamp("2024-01-01"), 30, 10, cfg, seed=1)
df_c = simulate_one_realization(pd.Timestamp("2024-01-01"), 30, 10, cfg, seed=2)
identical = bool((df_a["power_kw"].values == df_b["power_kw"].values).all())
diff_max = float(np.abs(df_a["power_kw"].values - df_c["power_kw"].values).max())
print(f"  seed=1 twice -> identical: {identical}")
print(f"  seed=1 vs seed=2 -> max diff: {diff_max:.2f} kW")
assert identical and diff_max > 0

print()
print("=" * 70)
print("Test 7 — Lighting capped at installed capacity")
print("=" * 70)
length_km = cfg.length_m / 1000
total_lanes = cfg.n_tubes * cfg.n_lanes_per_tube
installed = LIGHTING_PARAMS[cfg.lighting_type][0] * length_km * total_lanes
observed = float(df["lighting_kw"].max())
print(f"  Installed: {installed:.1f} kW  Max observed: {observed:.1f} kW  "
      f"-> {'CAPPED ✓' if observed <= installed + 0.01 else 'OVERSHOOT ✗'}")
assert observed <= installed + 0.01

print()
print("=" * 70)
print("Test 8 — Full Monte Carlo (20 runs, 1 year)")
print("=" * 70)
t0 = time.time()
results = run_monte_carlo(cfg, pd.Timestamp("2024-01-01"), 365, 10, n_runs=20, base_seed=42)
dt = time.time() - t0
print(f"  20 runs × 365 days × 10-min done in {dt:.2f} s")
kpis = results["kpis"]
print(f"  Annual energy MWh:  median={kpis['annualized_mwh'].median():.1f}  "
      f"P10={kpis['annualized_mwh'].quantile(0.1):.1f}  "
      f"P90={kpis['annualized_mwh'].quantile(0.9):.1f}")
print(f"  Peak power kW:      median={kpis['peak_kw'].median():.0f}  "
      f"P10={kpis['peak_kw'].quantile(0.1):.0f}  "
      f"P90={kpis['peak_kw'].quantile(0.9):.0f}")
print(f"  Envelope rows: {len(results['envelope'])}  "
      f"(P10..P90 spread mean: {(results['envelope']['p90']-results['envelope']['p10']).mean():.1f} kW)")

print()
print("=" * 70)
print("All tests passed ✓")
print("=" * 70)
