"""
app.py — Streamlit UI for the synthetic tunnel-load generator.

The simulation engine lives in simulator.py (pure NumPy/pandas, no UI).
This file is purely the visualization / interaction layer.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from simulator import (
    CONTEXT_MULT,
    LIGHTING_PARAMS,
    VENTILATION_PARAMS,
    TunnelConfig,
    run_monte_carlo,
)

# =============================================================================
# Page configuration
# =============================================================================

st.set_page_config(
    page_title="Tunnel Load Simulator",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =============================================================================
# Cached Monte Carlo wrapper
# =============================================================================

@st.cache_data(show_spinner=False)
def _cached_run(
    cfg_dict: dict,
    start_date_iso: str,
    n_days: int,
    freq_minutes: int,
    n_runs: int,
    base_seed: int,
) -> dict:
    cfg = TunnelConfig(**cfg_dict)
    return run_monte_carlo(
        cfg=cfg,
        start_date=pd.Timestamp(start_date_iso),
        n_days=n_days,
        freq_minutes=freq_minutes,
        n_runs=n_runs,
        base_seed=base_seed,
    )


# =============================================================================
# Plot helpers
# =============================================================================

def _downsample_hourly(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    return (
        df.set_index("timestamp")[cols]
        .resample("1h")
        .mean()
        .reset_index()
    )


def plot_decomposition(df: pd.DataFrame) -> go.Figure:
    plot_df = _downsample_hourly(
        df, ["lighting_kw", "ventilation_kw", "auxiliary_kw", "power_kw"]
    )
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=plot_df["timestamp"], y=plot_df["lighting_kw"],
        name="Lighting", stackgroup="comp",
        line=dict(width=0), fillcolor="rgba(255,193,7,0.55)",
    ))
    fig.add_trace(go.Scatter(
        x=plot_df["timestamp"], y=plot_df["ventilation_kw"],
        name="Ventilation", stackgroup="comp",
        line=dict(width=0), fillcolor="rgba(33,150,243,0.55)",
    ))
    fig.add_trace(go.Scatter(
        x=plot_df["timestamp"], y=plot_df["auxiliary_kw"],
        name="Auxiliary", stackgroup="comp",
        line=dict(width=0), fillcolor="rgba(120,120,120,0.55)",
    ))
    fig.add_trace(go.Scatter(
        x=plot_df["timestamp"], y=plot_df["power_kw"],
        name="Total power", line=dict(color="black", width=1.4),
    ))
    fig.update_layout(
        height=420, margin=dict(l=10, r=10, t=30, b=10),
        xaxis_title="Time", yaxis_title="Power [kW]",
        hovermode="x unified",
        legend=dict(orientation="h", y=1.08, x=0),
    )
    return fig


def plot_envelope(envelope: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=envelope["timestamp"], y=envelope["p90"],
        line=dict(width=0), showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=envelope["timestamp"], y=envelope["p10"],
        line=dict(width=0), fill="tonexty",
        fillcolor="rgba(63, 81, 181, 0.22)",
        name="P10–P90 band",
    ))
    fig.add_trace(go.Scatter(
        x=envelope["timestamp"], y=envelope["median"],
        line=dict(color="rgb(40, 60, 160)", width=1.5),
        name="Median",
    ))
    fig.add_trace(go.Scatter(
        x=envelope["timestamp"], y=envelope["mean"],
        line=dict(color="rgb(220, 80, 60)", width=1.0, dash="dot"),
        name="Mean",
    ))
    fig.update_layout(
        height=400, margin=dict(l=10, r=10, t=30, b=10),
        xaxis_title="Time", yaxis_title="Power [kW]",
        hovermode="x unified",
        legend=dict(orientation="h", y=1.08, x=0),
    )
    return fig


def plot_seasonal_profiles(season_long: pd.DataFrame) -> go.Figure:
    season_order = ["winter", "spring", "summer", "autumn"]
    color_map = {
        "winter": ("rgb(63,81,181)",  "rgba(63,81,181,0.18)"),
        "spring": ("rgb(76,175,80)",  "rgba(76,175,80,0.18)"),
        "summer": ("rgb(255,152,0)",  "rgba(255,152,0,0.18)"),
        "autumn": ("rgb(183,28,28)",  "rgba(183,28,28,0.18)"),
    }
    fig = go.Figure()
    for s in season_order:
        sub = season_long[season_long["season"] == s]
        if sub.empty:
            continue
        agg = sub.groupby("hour_int")["power_kw"].agg(["mean", "min", "max"]).reset_index()
        line_color, fill_color = color_map[s]
        fig.add_trace(go.Scatter(
            x=agg["hour_int"], y=agg["max"],
            line=dict(width=0), showlegend=False, hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(
            x=agg["hour_int"], y=agg["min"],
            line=dict(width=0), fill="tonexty",
            fillcolor=fill_color, showlegend=False, hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(
            x=agg["hour_int"], y=agg["mean"],
            name=s.capitalize(),
            line=dict(color=line_color, width=2.2),
            mode="lines+markers",
            marker=dict(size=5),
        ))
    fig.update_layout(
        height=420, margin=dict(l=10, r=10, t=30, b=10),
        xaxis=dict(
            title="Hour of day",
            dtick=2, tickmode="linear",
            range=[-0.3, 23.3],
        ),
        yaxis_title="Mean power [kW]",
        hovermode="x unified",
        legend=dict(orientation="h", y=1.08, x=0),
    )
    return fig


def plot_kpi_distributions(kpis_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    series = [
        ("annualized_mwh",      "Annual energy [MWh]",   "rgb(63,81,181)"),
        ("peak_kw",             "Peak power [kW]",       "rgb(255,152,0)"),
        ("specific_kwh_m_year", "Specific [kWh/m/y]",    "rgb(76,175,80)"),
        ("load_factor",         "Load factor",           "rgb(183,28,28)"),
    ]
    for col, name, color in series:
        fig.add_trace(go.Box(
            y=kpis_df[col], name=name,
            boxpoints="all", jitter=0.4, pointpos=0,
            marker=dict(color=color, size=4),
            line=dict(color=color),
        ))
    fig.update_layout(
        height=400, margin=dict(l=10, r=10, t=30, b=10),
        showlegend=False,
        yaxis_title="Value (mixed scales)",
    )
    return fig


# =============================================================================
# Sidebar (inputs)
# =============================================================================

st.title("⚡ Synthetic Tunnel Load Generator")
st.caption(
    "Vectorized Monte Carlo simulator — lighting, ventilation, auxiliary loads, "
    "traffic peaks, pollution & accident events."
)

with st.sidebar:
    st.header("⏱ Time horizon")
    start_date = st.date_input("Start date", value=pd.Timestamp("2024-01-01"))
    n_days = st.slider("Number of days", 7, 365 * 3, 365, step=7)
    freq_minutes = st.selectbox("Time step [min]", [5, 10, 15, 30, 60], index=1)

    st.header("🎲 Monte Carlo")
    n_runs = st.slider("Number of MC realizations", 1, 50, 20)
    base_seed = st.number_input("Base seed", 0, 10_000_000, 42)

    st.header("📐 Geometry")
    length_m = st.slider("Tunnel length [m]", 100, 12000, 1500, 100)
    n_tubes = st.selectbox("Number of tubes", [1, 2, 3, 4], index=1)
    n_lanes_per_tube = st.selectbox("Lanes per tube", [1, 2, 3, 4], index=1)
    altitude_m = st.slider("Altitude [m]", 0, 3000, 300, 50)
    max_depth_m = st.slider("Max depth / overburden [m]", 0, 1000, 80, 10)
    gradient_percent = st.slider("Mean gradient [%]", 0.0, 12.0, 2.0, 0.5)

    st.header("🛠 Operating context")
    tunnel_context = st.selectbox("Context", list(CONTEXT_MULT.keys()), index=1)
    lighting_type = st.selectbox("Lighting", list(LIGHTING_PARAMS.keys()), index=0)
    ventilation_type = st.selectbox("Ventilation", list(VENTILATION_PARAMS.keys()), index=1)
    aux_kw_per_km_tube = st.slider("Auxiliary load [kW/km/tube]", 5.0, 120.0, 35.0, 5.0)
    base_fixed_kw = st.slider("Fixed non-scaled load [kW]", 0.0, 500.0, 40.0, 10.0)

    st.header("🚗 Traffic pattern")
    traffic_level = st.slider("Global traffic level", 0.30, 2.00, 1.00, 0.05)
    morning_peak_hour = st.slider("Morning peak hour", 5.0, 11.0, 8.0, 0.25)
    evening_peak_hour = st.slider("Evening peak hour", 15.0, 22.0, 18.0, 0.25)
    peak_width_h = st.slider("Peak width [h]", 0.5, 4.0, 1.4, 0.1)
    traffic_sensitivity = st.slider("Ventilation traffic sensitivity", 0.05, 1.50, 0.65, 0.05)

    st.header("🌫 Stochastic effects")
    noise_sigma = st.slider("Gaussian noise sigma", 0.00, 0.50, 0.06, 0.01)
    pollution_probability = st.slider("Pollution event prob/day", 0.00, 0.50, 0.05, 0.01)
    accident_probability = st.slider("Accident prob/day", 0.00, 0.20, 0.015, 0.005)
    pollution_sensitivity = st.slider("Pollution vent multiplier", 0.00, 2.00, 0.55, 0.05)
    accident_sensitivity = st.slider("Accident vent multiplier", 0.00, 2.00, 0.75, 0.05)

cfg_dict = dict(
    length_m=float(length_m),
    n_tubes=int(n_tubes),
    n_lanes_per_tube=int(n_lanes_per_tube),
    altitude_m=float(altitude_m),
    max_depth_m=float(max_depth_m),
    gradient_percent=float(gradient_percent),
    tunnel_context=tunnel_context,
    lighting_type=lighting_type,
    ventilation_type=ventilation_type,
    aux_kw_per_km_tube=float(aux_kw_per_km_tube),
    base_fixed_kw=float(base_fixed_kw),
    traffic_level=float(traffic_level),
    morning_peak_hour=float(morning_peak_hour),
    evening_peak_hour=float(evening_peak_hour),
    peak_width_h=float(peak_width_h),
    traffic_sensitivity=float(traffic_sensitivity),
    noise_sigma=float(noise_sigma),
    pollution_probability_per_day=float(pollution_probability),
    accident_probability_per_day=float(accident_probability),
    pollution_sensitivity=float(pollution_sensitivity),
    accident_sensitivity=float(accident_sensitivity),
)

with st.spinner(f"Running {n_runs} Monte Carlo realization(s)…"):
    results = _cached_run(
        cfg_dict,
        pd.Timestamp(start_date).isoformat(),
        int(n_days),
        int(freq_minutes),
        int(n_runs),
        int(base_seed),
    )

representative = results["representative"]
kpis = results["kpis"]
envelope = results["envelope"]
season_profiles = results["season_profiles"]


# =============================================================================
# KPI summary
# =============================================================================

def _fmt_mc(values: pd.Series, fmt: str = "{:,.1f}") -> str:
    if len(values) == 1:
        return fmt.format(values.iloc[0])
    med = values.median()
    p10 = values.quantile(0.10)
    p90 = values.quantile(0.90)
    return f"{fmt.format(med)}  [{fmt.format(p10)} – {fmt.format(p90)}]"


st.subheader("📊 Key performance indicators")

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Annual energy [MWh]",     _fmt_mc(kpis["annualized_mwh"]))
c2.metric("Peak power [kW]",         _fmt_mc(kpis["peak_kw"], "{:,.0f}"))
c3.metric("Mean power [kW]",         _fmt_mc(kpis["mean_kw"], "{:,.0f}"))
c4.metric("Load factor",             _fmt_mc(kpis["load_factor"], "{:.2f}"))
c5.metric("Specific [kWh/m/year]",   _fmt_mc(kpis["specific_kwh_m_year"], "{:,.0f}"))

if n_runs > 1:
    st.caption(f"Format: **median  [P10 – P90]** across {n_runs} Monte Carlo realizations.")
else:
    st.caption("Single realization — increase MC runs to see uncertainty.")


# =============================================================================
# Tabs
# =============================================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🟢 One realization",
    "🌫 MC envelope",
    "🗓 Seasonal profiles",
    "📈 KPI distributions",
    "📋 Data preview",
])

with tab1:
    st.markdown(
        "Stacked components from **run #0** (representative realization). "
        "Hourly aggregation for plotting performance — full resolution preserved in CSV."
    )
    st.plotly_chart(plot_decomposition(representative), use_container_width=True)

with tab2:
    if n_runs >= 2:
        st.markdown(
            "**Median + P10–P90 envelope** of total power, hourly resampled, "
            "across all MC realizations."
        )
        st.plotly_chart(plot_envelope(envelope), use_container_width=True)
    else:
        st.info("Set Monte Carlo runs ≥ 2 to display the uncertainty envelope.")

with tab3:
    st.markdown(
        "**Mean daily profile per season** with **min–max band across MC runs** (shaded). "
        "Hour 0–23 on x-axis."
    )
    st.plotly_chart(plot_seasonal_profiles(season_profiles), use_container_width=True)

with tab4:
    if n_runs >= 2:
        st.markdown("**KPI distributions** across Monte Carlo realizations.")
        st.plotly_chart(plot_kpi_distributions(kpis), use_container_width=True)
        with st.expander("Per-run KPI table"):
            st.dataframe(kpis, use_container_width=True, hide_index=True)
    else:
        st.info("Set Monte Carlo runs ≥ 2 to display KPI distributions.")

with tab5:
    st.markdown("First 200 rows of run #0:")
    preview_cols = [
        "timestamp", "season", "day_type", "traffic_index",
        "pollution_event", "accident_event",
        "lighting_kw", "ventilation_kw", "auxiliary_kw",
        "power_kw", "energy_kwh",
    ]
    st.dataframe(
        representative[preview_cols].head(200),
        use_container_width=True, hide_index=True,
    )


# =============================================================================
# Export
# =============================================================================

st.subheader("💾 Export")

c_dl1, c_dl2 = st.columns(2)

with c_dl1:
    export_df = representative.assign(
        length_m=cfg_dict["length_m"],
        n_tubes=cfg_dict["n_tubes"],
        n_lanes_total=cfg_dict["n_tubes"] * cfg_dict["n_lanes_per_tube"],
        altitude_m=cfg_dict["altitude_m"],
        max_depth_m=cfg_dict["max_depth_m"],
        gradient_percent=cfg_dict["gradient_percent"],
        tunnel_context=cfg_dict["tunnel_context"],
        lighting_type=cfg_dict["lighting_type"],
        ventilation_type=cfg_dict["ventilation_type"],
    )
    csv_repr = export_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "📥 Representative run (CSV)",
        csv_repr,
        file_name="tunnel_load_representative.csv",
        mime="text/csv",
        use_container_width=True,
    )

with c_dl2:
    csv_kpis = kpis.to_csv(index=False).encode("utf-8")
    st.download_button(
        "📥 Monte Carlo KPIs (CSV)",
        csv_kpis,
        file_name="tunnel_load_mc_kpis.csv",
        mime="text/csv",
        use_container_width=True,
    )

st.warning(
    "**Scientific status:** first-order synthetic generator. For engineering or "
    "publication use, calibrate specific lighting, ventilation and auxiliary "
    "parameters against measured consumption, traffic counts, tunnel geometry "
    "and operating rules."
)
