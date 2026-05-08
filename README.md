# Tunnel Load Simulator

A vectorized **Monte Carlo** generator of synthetic electrical load profiles for road tunnels. Streamlit application that produces realistic time series of total electrical power and energy demand based on tunnel geometry, operating context, traffic patterns, and stochastic disturbances (pollution episodes, accidents, sensor noise).

> **Status:** first-order synthetic generator. Suitable for prototyping, tooling demos, and pre-design sensitivity studies. **Calibrate against measured data** before any engineering or publication use.

---

## What's new in this version

- **Real Monte Carlo**: N independent realizations → distributions of KPIs (median / P10 / P90) and an uncertainty envelope on the time series. The previous version produced a single trajectory and reported deterministic numbers — that wasn't really Monte Carlo.
- **Fully vectorized** simulation engine (no Python day loop). 1 year at 10-min resolution simulates in ~150 ms; 20 MC runs in ~0.6 s.
- **Plotly visualizations** with controlled hourly downsampling for browser performance — no more locked-up tabs.
- **Streamlit `cache_data`** on the simulator so slider changes are responsive.
- **Bugs fixed** (see *Bug fixes* section below).
- **Code split** into a pure simulation module (`simulator.py`) and a slim UI layer (`app.py`), so the engine is testable and reusable.

---

## Repository layout

```
.
├── app.py            # Streamlit UI (imports the engine)
├── simulator.py      # Pure NumPy/pandas simulation engine
├── test_engine.py    # Standalone validation tests (no Streamlit)
├── requirements.txt
├── README.md
└── LICENSE
```

---

## Quickstart

### Local

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app opens at `http://localhost:8501`.

### Streamlit Community Cloud

Push the repository to GitHub and point Streamlit Cloud at `app.py`. The `requirements.txt` is sufficient.

### Programmatic use (no UI)

```python
import pandas as pd
from simulator import TunnelConfig, run_monte_carlo

cfg = TunnelConfig(
    length_m=1500, n_tubes=2, n_lanes_per_tube=2,
    altitude_m=300, max_depth_m=80, gradient_percent=2.0,
    tunnel_context="peri-urban", lighting_type="LED adaptive",
    ventilation_type="longitudinal",
    aux_kw_per_km_tube=35.0, base_fixed_kw=40.0,
    traffic_level=1.0, morning_peak_hour=8.0, evening_peak_hour=18.0,
    peak_width_h=1.4, traffic_sensitivity=0.65,
    noise_sigma=0.06,
    pollution_probability_per_day=0.05, accident_probability_per_day=0.015,
    pollution_sensitivity=0.55, accident_sensitivity=0.75,
)

results = run_monte_carlo(
    cfg, pd.Timestamp("2024-01-01"),
    n_days=365, freq_minutes=10,
    n_runs=20, base_seed=42,
)
print(results["kpis"].describe())
```

---

## Monte Carlo methodology

For each user-selected configuration, the simulator performs **N independent runs** with seeds `base_seed + i` for `i = 0..N-1`. Each run produces a full time series at the chosen resolution.

From the N runs, the engine returns:

- **Per-run KPIs** (`results["kpis"]`): annual energy [MWh], peak power [kW], mean power [kW], load factor, specific consumption [kWh/m/year], event counts.
- **Distributional KPIs**: the UI summarises each KPI as `median  [P10 – P90]`, computed across runs.
- **Uncertainty envelope** (`results["envelope"]`): hourly-resampled total power with `median`, `p10`, `p90`, `mean` across runs, plotted as a fan chart.
- **Seasonal profiles** (`results["season_profiles"]`): mean hourly profile per season, per run, plotted as 4 lines with a min–max band across runs.

This is a real Monte Carlo treatment of the stochastic component of the model: events (pollution, accidents) and noises are independent across runs, so the spread of the envelope and the spread of the KPIs reflect actual operational variability.

---

## Bug fixes vs the previous version

- **`traffic_level` was neutralized.** Previously, traffic was normalized by its own quantile, so `traffic_level` cancelled out of the numerator and denominator. Now normalization uses a **fixed reference 1-day profile** independent of the simulation, so `traffic_level` is a true scaling parameter.
- **Stochastic events were truncated at midnight.** Previously, an event starting at 18 h with 8 h duration would simply stop at 24:00. Now events that wrap past midnight correctly continue into the next calendar day.
- **Lighting could exceed installed capacity.** Now `lighting_factor = min(factor, 1.0)`.
- **`mixed` and `sodium fixed` lighting types had identical parameters.** Now four genuinely distinct lighting models.
- **The four seasonal profiles all looked identical and only showed 00:00–02:00.** Caused by a string-indexed `time_of_day` plotted in narrow Streamlit columns. Now uses integer hour 0–23 with Plotly axes.
- **Per-day Python loop on event masking** was the main performance bottleneck. Now O(n_steps) vectorized.
- **No caching.** Now `@st.cache_data` on the Monte Carlo wrapper.

---

## Modeling

### Total instantaneous power

$$
P(t) = (P_{\text{base}} + P_{\text{light}}(t) + P_{\text{vent}}(t) + P_{\text{aux}}(t)) \cdot (1 + \varepsilon(t))
$$

with $\varepsilon(t) \sim \mathcal{N}(0, \sigma^2)$ a multiplicative Gaussian noise.

### Lighting

$$
P_{\text{light}}(t) = P_{\text{installed}} \cdot \min\Big(1,\; f_{\min} + (1 - f_{\min}) \cdot (1 - D(t)) + k_{\text{traffic}} \cdot T(t)\Big)
$$

where $P_{\text{installed}} = p_{\text{spec}} \cdot L_{\text{km}} \cdot N_{\text{lanes}}$, $D(t) \in [0,1]$ is the daylight proxy (sigmoid product on sunrise/sunset), $T(t)$ is the traffic index, and the cap at 1.0 prevents the model from drawing more than the installed capacity.

### Ventilation

$$
P_{\text{vent}}(t) = q_{\text{vent}} \cdot L_{\text{km}} \cdot N_{\text{tubes}} \cdot (f_{\min}^{\text{vent}} + s_{\text{traffic}} \cdot T(t)) \cdot a_{\text{alt}} \cdot a_{\text{slope}} \cdot (1 + s_{\text{poll}} \cdot \mathbb{1}_{\text{poll}}(t) + s_{\text{acc}} \cdot \mathbb{1}_{\text{acc}}(t))
$$

where the altitude factor $a_{\text{alt}} = 1 + \max(h - 500, 0)/6000$ and slope factor $a_{\text{slope}} = 1 + |g|/20$.

### Auxiliary

$$
P_{\text{aux}}(t) = q_{\text{aux}} \cdot L_{\text{km}} \cdot N_{\text{tubes}} \cdot a_{\text{depth}} \cdot \eta(t)
$$

with $\eta(t) \sim \mathcal{N}(1, 0.025^2)$ and $a_{\text{depth}} = 1 + \min(d, 800)/6000$.

### Traffic

$$
T(t) = \frac{T_{\text{raw}}(t)}{T_{\text{ref}}} \cdot c_{\text{ctx}} \cdot s_{\text{season}} \cdot d_{\text{dow}} \cdot \alpha_{\text{level}} \cdot u_d
$$

where $T_{\text{raw}}(t)$ is the sum of a night floor and morning/evening Gaussian peaks, $T_{\text{ref}}$ is the 98th-percentile of a fixed 1-day reference profile, $c_{\text{ctx}}$ is the context multiplier (urban 1.20, peri-urban 1.00, rural 0.78), $s_{\text{season}}$ the seasonal factor, $d_{\text{dow}}$ the weekday/weekend factor, $\alpha_{\text{level}}$ the user-controlled `traffic_level`, and $u_d \sim \mathcal{N}(1, 0.06^2)$ a per-day random multiplier.

### Stochastic events

For each calendar day $d$, two independent Bernoulli draws decide whether a pollution episode and/or an accident occur. If an event occurs, its start time and duration are drawn uniformly:

| Event type | start hour    | duration |
|-----------:|--------------:|---------:|
| pollution  | $\mathcal{U}(7, 18)$ | $\mathcal{U}(2, 8)$  |
| accident   | $\mathcal{U}(6.5, 20)$ | $\mathcal{U}(0.5, 3)$ |

Events that extend past 24:00 correctly spill into the next calendar day.

---

## Outputs

The Streamlit app exposes four visualization tabs and one data preview:

1. **One realization** — stacked decomposition (lighting / ventilation / auxiliary / total) of the representative run, hourly-aggregated for plotting.
2. **MC envelope** — fan chart with median + P10–P90 band of the total power across runs.
3. **Seasonal profiles** — mean daily profile per season (4 lines) with min–max bands across runs.
4. **KPI distributions** — box plots of annual energy, peak power, specific consumption, load factor across runs.

CSV exports:
- **Representative run**: full-resolution time series of run #0 with config metadata.
- **Monte Carlo KPIs**: one row per run.

---

## Calibration guidance

The default coefficients are first-order placeholders. To calibrate against a real tunnel, the most impactful parameters are:

- `aux_kw_per_km_tube`, `base_fixed_kw` — direct from monthly bills outside operating events.
- Lighting `(p_spec, k_traffic, f_min)` — from photometric design + measured minimum night demand.
- Ventilation `q_vent`, `f_min_vent` — from fan nameplate × duty cycle.
- Traffic peaks and `peak_width_h` — from loop counters or counts on the road approaches.
- Event probabilities — from operations log over 1–2 years.

For altitude correction, replace the linear term with `exp(h/8400)` (barometric scale height) if working at high elevation. For pollution event statistics, a Markov persistence model (probability of recurrence given a previous-day event) is more realistic than independent daily Bernoulli draws.

---

## License

MIT.
