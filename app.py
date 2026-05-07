import numpy as np
import pandas as pd
import streamlit as st
from dataclasses import dataclass

# ============================================================
# Scientific synthetic electrical-load generator for road tunnels
# Streamlit app - local execution
# ============================================================
# Model philosophy:
# - geometry-driven first-order scaling: length, tubes, lanes, altitude, depth, slope
# - operational decomposition: lighting + ventilation + auxiliary equipment + events
# - traffic-driven daily structure with configurable morning/evening peaks
# - stochastic Monte Carlo layer for plausible variability
# - not calibrated by default: parameters must be adjusted when measured data exist
# ============================================================

st.set_page_config(page_title="Tunnel load generator", layout="wide")

SEASON_MONTHS = {
    "winter": [12, 1, 2],
    "spring": [3, 4, 5],
    "summer": [6, 7, 8],
    "autumn": [9, 10, 11],
}


def get_season(month: int) -> str:
    for season, months in SEASON_MONTHS.items():
        if month in months:
            return season
    return "unknown"


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def gaussian_peak(hour, center, width, amplitude):
    return amplitude * np.exp(-0.5 * ((hour - center) / width) ** 2)


def daylight_proxy(hour, season):
    """Simple daylight proxy in [0, 1]. Demonstration-level, not astronomical."""
    params = {
        "winter": (8.0, 17.0),
        "spring": (6.5, 20.0),
        "summer": (5.5, 21.5),
        "autumn": (7.0, 18.5),
    }
    sunrise, sunset = params[season]
    return sigmoid((hour - sunrise) * 3.0) * sigmoid((sunset - hour) * 3.0)


def seasonal_factor(season):
    return {"winter": 1.12, "spring": 0.98, "summer": 0.92, "autumn": 1.03}[season]


def weekday_factor(dayofweek):
    if dayofweek < 5:
        return 1.00
    if dayofweek == 5:
        return 0.86
    return 0.78


def lighting_specific_kw_per_km_lane(lighting_type):
    # Approximate engineering scaling. To be calibrated with real tunnel data.
    return {
        "LED adaptive": 16.0,
        "LED fixed": 22.0,
        "mixed": 28.0,
        "sodium fixed": 35.0,
    }[lighting_type]


def ventilation_specific_kw_per_km_tube(ventilation_type):
    # Approximate installed/demand scaling. To be calibrated.
    return {
        "natural/low ventilation": 45.0,
        "longitudinal": 120.0,
        "semi-transverse": 200.0,
        "transverse": 320.0,
    }[ventilation_type]


@dataclass
class TunnelConfig:
    length_m: float
    n_tubes: int
    n_lanes_per_tube: int
    altitude_m: float
    max_depth_m: float
    gradient_percent: float
    tunnel_context: str
    lighting_type: str
    ventilation_type: str
    equipment_specific_kw_per_km: float
    base_fixed_kw: float
    traffic_level: float
    morning_peak_hour: float
    evening_peak_hour: float
    peak_width_h: float
    traffic_sensitivity: float
    pollution_sensitivity: float
    accident_sensitivity: float
    noise_sigma: float
    pollution_probability_per_day: float
    accident_probability_per_day: float
    seed: int


def simulate_tunnel_load(start_date, n_days, freq_minutes, cfg: TunnelConfig):
    rng = np.random.default_rng(cfg.seed)

    n_points = int(n_days * 24 * 60 / freq_minutes)
    idx = pd.date_range(start=pd.Timestamp(start_date), periods=n_points, freq=f"{freq_minutes}min")
    df = pd.DataFrame({"timestamp": idx})
    df["hour"] = df["timestamp"].dt.hour + df["timestamp"].dt.minute / 60.0
    df["dayofweek"] = df["timestamp"].dt.dayofweek
    df["season"] = df["timestamp"].dt.month.map(get_season)
    df["day_type"] = np.where(df["dayofweek"] < 5, "weekday", "weekend")

    length_km = cfg.length_m / 1000.0
    total_lanes = cfg.n_tubes * cfg.n_lanes_per_tube

    # Geometry and environmental correction factors
    altitude_factor = 1.0 + max(cfg.altitude_m - 500.0, 0.0) / 6000.0
    depth_factor = 1.0 + min(cfg.max_depth_m, 800.0) / 6000.0
    slope_factor = 1.0 + abs(cfg.gradient_percent) / 20.0

    # Daily stochastic multiplier
    day_index = df["timestamp"].dt.date
    unique_days = pd.Index(pd.unique(day_index))
    daily_random = pd.Series(rng.normal(1.0, 0.06, len(unique_days)), index=unique_days)
    df["daily_random"] = day_index.map(daily_random).astype(float)

    # Traffic profile
    morning = gaussian_peak(df["hour"].values, cfg.morning_peak_hour, cfg.peak_width_h, 1.00)
    evening = gaussian_peak(df["hour"].values, cfg.evening_peak_hour, cfg.peak_width_h * 1.15, 1.10)
    night_floor = 0.16 + 0.06 * np.cos(2 * np.pi * (df["hour"].values - 3.0) / 24.0)

    raw_traffic = night_floor + morning + evening
    if cfg.tunnel_context == "urban":
        context_factor = 1.20
        raw_traffic += gaussian_peak(df["hour"].values, 18.7, 2.5, 0.25)
    elif cfg.tunnel_context == "rural":
        context_factor = 0.78
    else:
        context_factor = 1.00

    day_factor = df["dayofweek"].map(weekday_factor).astype(float).values
    season_factor = df["season"].map(seasonal_factor).astype(float).values

    traffic = (
        raw_traffic
        * day_factor
        * season_factor
        * context_factor
        * cfg.traffic_level
        * df["daily_random"].values
    )
    traffic = traffic / max(np.quantile(traffic, 0.98), 1e-6)
    traffic = np.clip(traffic, 0.0, 1.5)
    df["traffic_index"] = traffic

    # Daylight proxy
    daylight = np.zeros(len(df))
    for season in ["winter", "spring", "summer", "autumn"]:
        mask = df["season"].values == season
        daylight[mask] = daylight_proxy(df.loc[mask, "hour"].values, season)

    # Lighting: length x lanes x technology, modulated by daylight and traffic
    lighting_installed_kw = lighting_specific_kw_per_km_lane(cfg.lighting_type) * length_km * total_lanes
    if cfg.lighting_type == "LED adaptive":
        traffic_coupling = 0.18
        minimum_fraction = 0.20
    elif cfg.lighting_type == "LED fixed":
        traffic_coupling = 0.06
        minimum_fraction = 0.28
    else:
        traffic_coupling = 0.04
        minimum_fraction = 0.35

    night_component = 1.0 - daylight
    lighting_kw = lighting_installed_kw * (
        minimum_fraction + (1.0 - minimum_fraction) * night_component + traffic_coupling * traffic
    )

    # Ventilation: length x tubes x ventilation type, driven by traffic, pollution, accidents, altitude, slope
    ventilation_scale_kw = ventilation_specific_kw_per_km_tube(cfg.ventilation_type) * length_km * cfg.n_tubes
    ventilation_kw = ventilation_scale_kw * (
        0.15 + cfg.traffic_sensitivity * traffic
    ) * altitude_factor * slope_factor

    # Events
    pollution_event = np.zeros(len(df), dtype=int)
    accident_event = np.zeros(len(df), dtype=int)

    for day in unique_days:
        if rng.random() < cfg.pollution_probability_per_day:
            start_h = rng.uniform(7.0, 18.0)
            duration_h = rng.uniform(2.0, 8.0)
            mask = (day_index == day) & (df["hour"] >= start_h) & (df["hour"] <= start_h + duration_h)
            pollution_event[mask.values] = 1

        if rng.random() < cfg.accident_probability_per_day:
            start_h = rng.uniform(6.5, 20.0)
            duration_h = rng.uniform(0.5, 3.0)
            mask = (day_index == day) & (df["hour"] >= start_h) & (df["hour"] <= start_h + duration_h)
            accident_event[mask.values] = 1

    df["pollution_event"] = pollution_event
    df["accident_event"] = accident_event

    ventilation_kw *= (
        1.0
        + cfg.pollution_sensitivity * pollution_event
        + cfg.accident_sensitivity * accident_event
    )

    # Auxiliary load: safety systems, cameras, IT, pumping, monitoring
    auxiliary_kw = cfg.equipment_specific_kw_per_km * length_km * cfg.n_tubes * depth_factor
    auxiliary_kw = auxiliary_kw * rng.normal(1.0, 0.025, len(df))

    deterministic_kw = cfg.base_fixed_kw + lighting_kw + ventilation_kw + auxiliary_kw
    power_kw = np.maximum(deterministic_kw * (1.0 + rng.normal(0.0, cfg.noise_sigma, len(df))), 0.0)

    df["length_m"] = cfg.length_m
    df["n_tubes"] = cfg.n_tubes
    df["n_lanes_total"] = total_lanes
    df["altitude_m"] = cfg.altitude_m
    df["max_depth_m"] = cfg.max_depth_m
    df["gradient_percent"] = cfg.gradient_percent
    df["tunnel_context"] = cfg.tunnel_context
    df["lighting_type"] = cfg.lighting_type
    df["ventilation_type"] = cfg.ventilation_type
    df["lighting_kw"] = lighting_kw
    df["ventilation_kw"] = ventilation_kw
    df["auxiliary_kw"] = auxiliary_kw
    df["power_kW"] = power_kw
    df["energy_kWh"] = power_kw * freq_minutes / 60.0

    return df


# ============================================================
# Interface
# ============================================================

st.title("Synthetic electrical-load generator for road tunnels")
st.caption("Geometry-driven Monte Carlo simulator: lighting, ventilation, auxiliary loads, traffic peaks, pollution and accident events.")

with st.sidebar:
    st.header("Time horizon")
    start_date = st.date_input("Start date", value=pd.Timestamp("2024-01-01"))
    n_days = st.slider("Number of days", min_value=7, max_value=365 * 3, value=365 * 2, step=7)
    freq_minutes = st.selectbox("Time step [min]", [5, 10, 15, 30, 60], index=1)
    seed = st.number_input("Monte Carlo seed", 0, 10_000_000, 42, 1)

    st.header("Geometry")
    length_m = st.slider("Tunnel length [m]", 100, 12000, 1500, 100)
    n_tubes = st.selectbox("Number of tubes", [1, 2, 3, 4], index=1)
    n_lanes_per_tube = st.selectbox("Lanes per tube", [1, 2, 3, 4], index=1)
    altitude_m = st.slider("Altitude [m]", 0, 3000, 300, 50)
    max_depth_m = st.slider("Maximum depth / overburden proxy [m]", 0, 1000, 80, 10)
    gradient_percent = st.slider("Mean absolute gradient [%]", 0.0, 12.0, 2.0, 0.5)

    st.header("Operating context")
    tunnel_context = st.selectbox("Context", ["urban", "peri-urban", "rural"], index=1)
    lighting_type = st.selectbox("Lighting", ["LED adaptive", "LED fixed", "mixed", "sodium fixed"], index=0)
    ventilation_type = st.selectbox(
        "Ventilation",
        ["natural/low ventilation", "longitudinal", "semi-transverse", "transverse"],
        index=1,
    )
    equipment_specific_kw_per_km = st.slider("Auxiliary load [kW/km/tube]", 5.0, 120.0, 35.0, 5.0)
    base_fixed_kw = st.slider("Fixed non-scaled load [kW]", 0.0, 500.0, 40.0, 10.0)

    st.header("Traffic pattern")
    traffic_level = st.slider("Global traffic level", 0.30, 2.00, 1.00, 0.05)
    morning_peak_hour = st.slider("Morning peak hour", 5.0, 11.0, 8.0, 0.25)
    evening_peak_hour = st.slider("Evening peak hour", 15.0, 22.0, 18.0, 0.25)
    peak_width_h = st.slider("Peak width [h]", 0.5, 4.0, 1.4, 0.1)
    traffic_sensitivity = st.slider("Ventilation traffic sensitivity", 0.05, 1.50, 0.65, 0.05)

    st.header("Stochastic effects")
    noise_sigma = st.slider("Gaussian multiplicative noise sigma", 0.00, 0.50, 0.06, 0.01)
    pollution_probability = st.slider("Pollution-event probability per day", 0.00, 0.50, 0.05, 0.01)
    accident_probability = st.slider("Accident probability per day", 0.00, 0.20, 0.015, 0.005)
    pollution_sensitivity = st.slider("Pollution ventilation multiplier", 0.00, 2.00, 0.55, 0.05)
    accident_sensitivity = st.slider("Accident ventilation multiplier", 0.00, 2.00, 0.75, 0.05)

cfg = TunnelConfig(
    length_m=length_m,
    n_tubes=n_tubes,
    n_lanes_per_tube=n_lanes_per_tube,
    altitude_m=altitude_m,
    max_depth_m=max_depth_m,
    gradient_percent=gradient_percent,
    tunnel_context=tunnel_context,
    lighting_type=lighting_type,
    ventilation_type=ventilation_type,
    equipment_specific_kw_per_km=equipment_specific_kw_per_km,
    base_fixed_kw=base_fixed_kw,
    traffic_level=traffic_level,
    morning_peak_hour=morning_peak_hour,
    evening_peak_hour=evening_peak_hour,
    peak_width_h=peak_width_h,
    traffic_sensitivity=traffic_sensitivity,
    pollution_sensitivity=pollution_sensitivity,
    accident_sensitivity=accident_sensitivity,
    noise_sigma=noise_sigma,
    pollution_probability_per_day=pollution_probability,
    accident_probability_per_day=accident_probability,
    seed=int(seed),
)

if st.button("Run Monte Carlo simulation", type="primary") or "df" not in st.session_state:
    st.session_state["df"] = simulate_tunnel_load(start_date, n_days, freq_minutes, cfg)

df = st.session_state["df"]

# Metrics
annualized_mwh = df["energy_kWh"].sum() / 1000.0 * 365.0 / n_days
total_mwh = df["energy_kWh"].sum() / 1000.0
peak_kw = df["power_kW"].max()
mean_kw = df["power_kW"].mean()
load_factor = mean_kw / peak_kw if peak_kw > 0 else np.nan
specific_kwh_m_year = annualized_mwh * 1000.0 / length_m

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total simulated energy", f"{total_mwh:,.1f} MWh")
c2.metric("Annualized energy", f"{annualized_mwh:,.1f} MWh/year")
c3.metric("Specific energy", f"{specific_kwh_m_year:,.0f} kWh/m/year")
c4.metric("Mean / peak power", f"{mean_kw:,.0f} / {peak_kw:,.0f} kW")
c5.metric("Load factor", f"{load_factor:.2f}")

st.subheader("Synthetic electrical load")
st.line_chart(df.set_index("timestamp")[["power_kW", "lighting_kw", "ventilation_kw", "auxiliary_kw"]])

st.subheader("Four seasonal mean daily profiles")
profile = (
    df.assign(time_of_day=df["timestamp"].dt.strftime("%H:%M"))
    .groupby(["season", "time_of_day"], as_index=False)["power_kW"]
    .mean()
)

cols = st.columns(4)
for i, season in enumerate(["winter", "spring", "summer", "autumn"]):
    with cols[i]:
        st.markdown(f"**{season.capitalize()}**")
        tmp = profile[profile["season"] == season].set_index("time_of_day")[["power_kW"]]
        st.line_chart(tmp)

st.subheader("Data preview")
export_cols = [
    "timestamp", "season", "day_type", "length_m", "n_tubes", "n_lanes_total",
    "altitude_m", "max_depth_m", "gradient_percent", "tunnel_context",
    "lighting_type", "ventilation_type", "traffic_index", "pollution_event",
    "accident_event", "lighting_kw", "ventilation_kw", "auxiliary_kw",
    "power_kW", "energy_kWh",
]
st.dataframe(df[export_cols].head(300), use_container_width=True)

csv = df[export_cols].to_csv(index=False).encode("utf-8")
st.download_button(
    "Download complete synthetic series as CSV",
    csv,
    file_name="synthetic_tunnel_load.csv",
    mime="text/csv",
)

st.warning(
    "Scientific status: this is a first-order synthetic generator. "
    "For publication or engineering use, calibrate the specific lighting, ventilation and auxiliary parameters "
    "against measured consumption, traffic counts, tunnel geometry and operating rules."
)

