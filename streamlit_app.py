import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import math
import lightgbm as lgb
from sklearn.multioutput import MultiOutputRegressor

st.set_page_config(page_title="Refinery Tank Farm & LightGBM CDU System", layout="wide")

# Tank Physical Constants (Specified Dimensions)
TANK_HEIGHT = 13.61        # meters
TANK_DIAMETER = 7.90       # meters inner diameter
TANK_RADIUS = TANK_DIAMETER / 2.0
TANK_AREA = np.pi * (TANK_RADIUS ** 2)  # ~49.017 m2
TANK_VOLUME = TANK_AREA * TANK_HEIGHT   # ~667.12 m3

# 10 Crude Assays Catalog (>8 Crudes)
CRUDE_CATALOG = {
    "Arab Light":       {"API": 33.4, "Sulfur": 1.77, "TAN": 0.15},
    "Arab Heavy":       {"API": 27.9, "Sulfur": 2.85, "TAN": 0.30},
    "Arab Extra Light": {"API": 39.4, "Sulfur": 1.09, "TAN": 0.08},
    "Bonny Light":      {"API": 35.3, "Sulfur": 0.14, "TAN": 0.28},
    "Maya":             {"API": 21.8, "Sulfur": 3.45, "TAN": 0.50},
    "Brent":            {"API": 38.3, "Sulfur": 0.37, "TAN": 0.10},
    "Basrah Medium":    {"API": 29.0, "Sulfur": 3.00, "TAN": 0.22},
    "Murban":           {"API": 40.5, "Sulfur": 0.78, "TAN": 0.05},
    "Urals":            {"API": 31.7, "Sulfur": 1.44, "TAN": 0.20},
    "Kuwait Export":    {"API": 30.2, "Sulfur": 2.70, "TAN": 0.18}
}

def api_to_density(api):
    sg = 141.5 / (api + 131.5)
    return sg * 1000.0  # kg/m3

def density_to_api(density_kg_m3):
    sg = density_kg_m3 / 1000.0
    return (141.5 / sg) - 131.5

# Cached LightGBM Multi-Output CDU Surrogate Model
@st.cache_resource
def get_or_train_cdu_model(num_trees=150, lr=0.05, leaves=31):
    np.random.seed(42)
    n_samples = 4000

    api = np.random.uniform(18.0, 42.0, n_samples)
    sulfur = np.random.uniform(0.1, 4.0, n_samples)
    tan = np.random.uniform(0.05, 1.5, n_samples)
    feed_rate = np.random.uniform(50.0, 500.0, n_samples)

    # Physical yield approximations (% wt)
    naphtha = np.clip(0.8 * api - 2.5 * sulfur + np.random.normal(0, 1.0, n_samples), 5, 38)
    kero = np.clip(0.3 * api + 1.2 * sulfur + np.random.normal(0, 0.7, n_samples), 6, 22)
    diesel = np.clip(35.0 - 0.25 * api - 1.0 * sulfur + np.random.normal(0, 0.9, n_samples), 15, 40)
    residue = np.clip(60.0 - 1.1 * api + 3.0 * sulfur + np.random.normal(0, 1.2, n_samples), 10, 55)
    lpg_offgas = np.clip(100.0 - (naphtha + kero + diesel + residue), 1.5, 8.0)

    # Normalize to 100%
    total = naphtha + kero + diesel + residue + lpg_offgas
    X = pd.DataFrame({"API": api, "Sulfur_wt": sulfur, "TAN": tan, "FeedRate_m3h": feed_rate})
    y = pd.DataFrame({
        "LPG_Offgas_wt%": (lpg_offgas / total) * 100,
        "Naphtha_wt%": (naphtha / total) * 100,
        "Kerosene_wt%": (kero / total) * 100,
        "Diesel_wt%": (diesel / total) * 100,
        "Residue_wt%": (residue / total) * 100
    })

    base_lgb = lgb.LGBMRegressor(
        n_estimators=num_trees,
        learning_rate=lr,
        num_leaves=leaves,
        random_state=42,
        verbose=-1
    )
    model = MultiOutputRegressor(base_lgb)
    model.fit(X, y)
    return model

# Global Session State
if "tanks_data" not in st.session_state:
    st.session_state.tanks_data = {
        f"Tank {i+1}": {
            "layers": [
                {"crude": "Maya", "tons": 60.0},
                {"crude": "Arab Heavy", "tons": 50.0},
                {"crude": "Urals", "tons": 55.0},
                {"crude": "Arab Light", "tons": 45.0},
                {"crude": "Basrah Medium", "tons": 40.0},
                {"crude": "Bonny Light", "tons": 35.0},
                {"crude": "Brent", "tons": 30.0},
                {"crude": "Murban", "tons": 25.0},
                {"crude": "Arab Extra Light", "tons": 15.0}
            ],
            "initial_flow": 20.0 if i < 2 else 0.0,
            "pump_active": True if i < 2 else False
        }
        for i in range(4)
    }

if "flow_events" not in st.session_state:
    st.session_state.flow_events = [
        {"time_offset_hrs": 6, "tank": "Tank 1", "new_flow": 35.0},
        {"time_offset_hrs": 12, "tank": "Tank 2", "new_flow": 10.0}
    ]

# Navigation Sidebar
with st.sidebar:
    st.title("🧭 Navigation")
    page = st.radio("Go to Page", ["1. Tank Farm & Blending Simulator", "2. LightGBM CDU Model Engine"])
    st.divider()

# ==========================================
# PAGE 1: TANK FARM & DYNAMIC BLENDING
# ==========================================
if page == "1. Tank Farm & Blending Simulator":
    st.title("🛢️ Tank Farm Inventory & Dynamic CDU Blending")
    st.caption(f"Geometry: Diameter = {TANK_DIAMETER} m (Radius {TANK_RADIUS:.2f} m) | Height = {TANK_HEIGHT} m | Volume = {TANK_VOLUME:.1f} m³")

    with st.sidebar:
        st.header("⏱️ Timeline & Kinetics")
        start_date = st.date_input("Start Date", value=datetime.now().date())
        start_time = st.time_input("Start Time", value=datetime.now().time())
        start_datetime = datetime.combine(start_date, start_time)

        sim_hours = st.slider("Simulation Duration (Hours)", 6, 72, 24)
        settling_tau = st.slider("Separation Time Constant (τ hrs)", 6.0, 48.0, 18.0,
                                 help="Kinetics of de-mixing after circulation is halted.")

        st.divider()
        st.header("🔀 Flow Adjustments")
        with st.expander("Add Flow Change Event"):
            ev_tank = st.selectbox("Tank", list(st.session_state.tanks_data.keys()))
            ev_hr = st.number_input("Hour Offset", 1, sim_hours - 1, 6)
            ev_flow = st.number_input("New Flow (m³/h)", 0.0, 100.0, 25.0)
            if st.button("Save Event"):
                st.session_state.flow_events.append({"time_offset_hrs": ev_hr, "tank": ev_tank, "new_flow": ev_flow})
                st.rerun()

        if st.session_state.flow_events:
            st.dataframe(pd.DataFrame(st.session_state.flow_events), use_container_width=True)
            if st.button("Clear All Events"):
                st.session_state.flow_events = []
                st.rerun()

    # Tank Farm Configuration Tabs
    st.subheader("Tank Layer Inventory & Booster Pump Control")
    t_tabs = st.tabs(list(st.session_state.tanks_data.keys()))

    for idx, (t_name, t_info) in enumerate(st.session_state.tanks_data.items()):
        with t_tabs[idx]:
            c1, c2 = st.columns([2, 3])
            with c1:
                t_info["pump_active"] = st.checkbox("Pump Active", value=t_info["pump_active"], key=f"pa_{t_name}")
                t_info["initial_flow"] = st.number_input("Initial Flow (m³/h)", 0.0, 100.0, float(t_info["initial_flow"]), key=f"if_{t_name}")

                st.markdown("**Add Crude Parcel Layer**")
                crude_sel = st.selectbox("Crude Type", list(CRUDE_CATALOG.keys()), key=f"cs_{t_name}")
                mass_val = st.number_input("Tons Added", 0.0, 500.0, 30.0, key=f"mv_{t_name}")
                if st.button("Add Parcel", key=f"ap_{t_name}"):
                    t_info["layers"].append({"crude": crude_sel, "tons": mass_val})
                    st.rerun()
                if st.button("Clear Layers", key=f"cl_{t_name}"):
                    t_info["layers"] = []
                    st.rerun()

            with c2:
                if t_info["layers"]:
                    inv_rows = []
                    total_v = 0.0
                    total_m = 0.0
                    for lyr in t_info["layers"]:
                        d = api_to_density(CRUDE_CATALOG[lyr["crude"]]["API"])
                        v = (lyr["tons"] * 1000.0) / d
                        total_v += v
                        total_m += lyr["tons"]
                        inv_rows.append({
                            "Crude": lyr["crude"],
                            "Tons": lyr["tons"],
                            "Vol (m³)": round(v, 2),
                            "API": CRUDE_CATALOG[lyr["crude"]]["API"],
                            "Sulfur (wt%)": CRUDE_CATALOG[lyr["crude"]]["Sulfur"]
                        })
                    st.dataframe(pd.DataFrame(inv_rows), height=190, use_container_width=True)
                    st.info(f"Inventory: **{total_v:.1f} m³** | Level: **{total_v/TANK_AREA:.2f} m** / {TANK_HEIGHT} m | Mass: **{total_m:.1f} t**")
                else:
                    st.warning("No parcels present.")

    # Simulation Execution
    sim_records = []
    manifold_records = []

    for t_idx in range(sim_hours + 1):
        step_time = start_datetime + timedelta(hours=t_idx)
        m_flow, m_mass, m_dens_vol, m_sulf_mass, m_tan_mass = 0.0, 0.0, 0.0, 0.0, 0.0

        for t_name, t_info in st.session_state.tanks_data.items():
            rate = t_info["initial_flow"] if t_info["pump_active"] else 0.0
            for ev in st.session_state.flow_events:
                if ev["tank"] == t_name and t_idx >= ev["time_offset_hrs"]:
                    rate = ev["new_flow"]

            cur_vols = {}
            for lyr in t_info["layers"]:
                c = lyr["crude"]
                d = api_to_density(CRUDE_CATALOG[c]["API"])
                v = (lyr["tons"] * 1000.0) / d
                cur_vols[c] = cur_vols.get(c, 0.0) + v

            tank_vol = sum(cur_vols.values())
            if tank_vol <= 0:
                rate = 0.0
                tank_vol = 0.0

            lvl = min(TANK_HEIGHT, tank_vol / TANK_AREA)

            # De-mixing kinetics
            if tank_vol > 0:
                mean_rho = sum(cur_vols[c] * api_to_density(CRUDE_CATALOG[c]["API"]) for c in cur_vols) / tank_vol
                demix_factor = (1.0 - np.exp(-t_idx / settling_tau))
                weights = {}
                for c, v in cur_vols.items():
                    rho = api_to_density(CRUDE_CATALOG[c]["API"])
                    bias = 1.0 + 1.2 * ((rho - mean_rho) / mean_rho) * demix_factor
                    weights[c] = max(0.0001, (v / tank_vol) * bias)
                norm = sum(weights.values())
                fractions = {c: weights[c] / norm for c in weights}
            else:
                fractions = {c: 0.0 for c in CRUDE_CATALOG}

            # Inventory withdrawal
            vol_draw = rate * 1.0
            if vol_draw > 0 and tank_vol > 0:
                for lyr in t_info["layers"]:
                    c = lyr["crude"]
                    d = api_to_density(CRUDE_CATALOG[c]["API"])
                    d_vol = min(vol_draw * fractions.get(c, 0.0), (lyr["tons"] * 1000.0) / d)
                    lyr["tons"] = max(0.0, lyr["tons"] - (d_vol * d / 1000.0))

            sim_records.append({
                "Timestamp": step_time,
                "Hour": t_idx,
                "Tank": t_name,
                "Level_m": lvl,
                "Volume_m3": tank_vol,
                "Flow_m3h": rate,
                **{f"frac_{k}": fractions.get(k, 0.0) for k in CRUDE_CATALOG}
            })

            if rate > 0 and tank_vol > 0:
                t_rho = sum(fractions[c] * api_to_density(CRUDE_CATALOG[c]["API"]) for c in fractions)
                t_m = rate * (t_rho / 1000.0)
                m_flow += rate
                m_mass += t_m
                m_dens_vol += rate * t_rho
                m_sulf_mass += t_m * sum(fractions[c] * CRUDE_CATALOG[c]["Sulfur"] for c in fractions)
                m_tan_mass += t_m * sum(fractions[c] * CRUDE_CATALOG[c]["TAN"] for c in fractions)

        if m_flow > 0:
            bl_dens = m_dens_vol / m_flow
            manifold_records.append({
                "Timestamp": step_time,
                "Hour": t_idx,
                "Flow_m3h": m_flow,
                "Mass_th": m_mass,
                "API": density_to_api(bl_dens),
                "Sulfur": m_sulf_mass / m_mass,
                "TAN": m_tan_mass / m_mass
            })
        else:
            manifold_records.append({
                "Timestamp": step_time,
                "Hour": t_idx,
                "Flow_m3h": 0.0, "Mass_th": 0.0, "API": np.nan, "Sulfur": np.nan, "TAN": np.nan
            })

    sim_df = pd.DataFrame(sim_records)
    manifold_df = pd.DataFrame(manifold_records)

    # Dynamic Charts
    st.divider()
    st.subheader("Dynamic Feed and Level Curves")
    gc1, gc2 = st.columns(2)

    with gc1:
        fig_lvl = go.Figure()
        for t_name in st.session_state.tanks_data.keys():
            sub = sim_df[sim_df["Tank"] == t_name]
            fig_lvl.add_trace(go.Scatter(x=sub["Timestamp"], y=sub["Level_m"], mode="lines", name=f"{t_name} Level"))
        fig_lvl.update_layout(title="Level Depletion vs. Time", xaxis_title="Date & Time", yaxis_title="Level (m)", yaxis_range=[0, TANK_HEIGHT + 1], hovermode="x unified")
        st.plotly_chart(fig_lvl, use_container_width=True)

    with gc2:
        fig_p = go.Figure()
        fig_p.add_trace(go.Scatter(x=manifold_df["Timestamp"], y=manifold_df["API"], name="API (°)", yaxis="y1", line=dict(color="orange")))
        fig_p.add_trace(go.Scatter(x=manifold_df["Timestamp"], y=manifold_df["Sulfur"], name="Sulfur (wt%)", yaxis="y2", line=dict(color="crimson")))
        fig_p.update_layout(title="CDU Header Properties Drift", xaxis_title="Date & Time", yaxis=dict(title="Blended API (°)"), yaxis2=dict(title="Sulfur (wt%)", overlaying="y", side="right"), hovermode="x unified")
        st.plotly_chart(fig_p, use_container_width=True)

    # Withdrawal Fraction Breakdown Table
    st.divider()
    st.subheader("Hourly Parcel Fraction Table (Total = 1.0)")
    hr_sel = st.slider("Inspect Simulation Hour", 0, sim_hours, 0)
    inspect_data = sim_df[sim_df["Hour"] == hr_sel].copy()

    f_cols = [c for c in inspect_data.columns if c.startswith("frac_") and inspect_data[c].sum() > 0]
    renames = {c: c.replace("frac_", "") for c in f_cols}
    show_df = inspect_data[["Tank", "Timestamp", "Flow_m3h", "Level_m", "Volume_m3"] + f_cols].rename(columns=renames)
    show_df["Total Fraction"] = show_df[list(renames.values())].sum(axis=1).round(4)
    st.dataframe(show_df.style.format({c: "{:.4f}" for c in list(renames.values()) + ["Total Fraction"]}), use_container_width=True)

    # Online CDU Cut Yield Prediction for Current Selected Hour
    cdu_model = get_or_train_cdu_model()
    curr_m = manifold_df[manifold_df["Hour"] == hr_sel].iloc[0]
    if curr_m["Flow_m3h"] > 0 and not np.isnan(curr_m["API"]):
        st.markdown(f"**CDU Yields at Hour {hr_sel}** (Feed: {curr_m['Flow_m3h']:.1f} m³/h | {curr_m['API']:.2f}° API | {curr_m['Sulfur']:.2f} wt% S)")
        pred_feats = pd.DataFrame([{"API": curr_m["API"], "Sulfur_wt": curr_m["Sulfur"], "TAN": curr_m["TAN"], "FeedRate_m3h": curr_m["Flow_m3h"]}])
        cuts_pred = cdu_model.predict(pred_feats)[0]
        cuts = ["LPG / Offgas", "Naphtha", "Kerosene / Jet", "Diesel", "Residue"]
        res_df = pd.DataFrame({"Cut": cuts, "Yield (wt%)": np.round(cuts_pred, 2), "Flow (tons/h)": np.round((cuts_pred / 100.0) * curr_m["Mass_th"], 2)})
        st.table(res_df)

# ==========================================
# PAGE 2: LIGHTGBM MODEL ENGINE
# ==========================================
elif page == "2. LightGBM CDU Model Engine":
    st.title("⚙️ LightGBM CDU Surrogate Model Engine")
    st.write("Inspect, tune, and test the LightGBM multi-target distillation column predictor.")

    m_col1, m_col2 = st.columns([1, 2])
    with m_col1:
        st.subheader("Hyperparameter Tuning")
        trees = st.slider("Trees (n_estimators)", 50, 300, 150, step=25)
        l_rate = st.select_slider("Learning Rate", options=[0.01, 0.03, 0.05, 0.1, 0.2], value=0.05)
        leaves = st.slider("Num Leaves", 15, 63, 31)

        if st.button("Retrain Model"):
            get_or_train_cdu_model.clear()
            st.success("Model retrained successfully!")

    with m_col2:
        st.subheader("Manual Feed Assay Inference")
        c_api = st.slider("Crude API Gravity (°)", 18.0, 42.0, 32.5)
        c_s = st.slider("Sulfur Content (wt%)", 0.1, 4.0, 1.8)
        c_tan = st.slider("Total Acid Number (mg KOH/g)", 0.05, 1.5, 0.25)
        c_rate = st.slider("Distillation Feed Rate (m³/h)", 50.0, 500.0, 200.0)

        trained_model = get_or_train_cdu_model(num_trees=trees, lr=l_rate, leaves=leaves)
        sample_in = pd.DataFrame([{"API": c_api, "Sulfur_wt": c_s, "TAN": c_tan, "FeedRate_m3h": c_rate}])
        preds = trained_model.predict(sample_in)[0]

        cuts = ["LPG / Offgas", "Naphtha", "Kerosene", "Diesel", "Residue"]
        out_df = pd.DataFrame({"Cut Fraction": cuts, "Predicted Yield (wt%)": np.round(preds, 2)})

        st.table(out_df)
        st.bar_chart(out_df.set_index("Cut Fraction")["Predicted Yield (wt%)"])
