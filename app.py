"""
app.py — Streamlit Dashboard for the Vayu-Rakshak Air Quality Monitoring System.

Four sidebar tabs:
  1. 🗺️  Dashboard        — Date-filtered Folium heatmap (PM2.5 intensity)
  2. 🚨  Surveillance     — KPI cards for anomalous / failed sensors
  3. ➕  Register Sensor  — Form to register new sensors via the FastAPI API
  4. 📈  Historical Analytics — Plotly time-series + CSV export

Persistent chatbot:
  "Dr. Vayu" LangChain SQL agent is always accessible via the chat panel
  in any tab. It queries the DB AND checks nearby POIs to explain readings.
"""

import os
import io
import re
import json
import logging
import requests
import pandas as pd
import streamlit as st
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium
import plotly.graph_objects as go
from datetime import datetime, timedelta, date

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
API_BASE = os.getenv("VAYU_API_BASE", "http://localhost:8000")

logging.basicConfig(level=logging.WARNING)

st.set_page_config(
    page_title="Vayu-Rakshak | Air Quality Monitor",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# Custom CSS — premium dark theme
# ─────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

  /* Sidebar */
  [data-testid="stSidebar"] {
    background: linear-gradient(160deg, #0f2027, #203a43, #2c5364);
    color: #e0f7fa;
  }
  [data-testid="stSidebar"] .stRadio label { color: #b2ebf2 !important; font-weight: 500; }

  /* Main background */
  .stApp { background: #0d1117; color: #e6edf3; }

  /* Metric cards */
  [data-testid="metric-container"] {
    background: linear-gradient(135deg, #1e2a38, #162032);
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 1rem;
    box-shadow: 0 4px 15px rgba(0,0,0,0.4);
  }
  [data-testid="metric-container"] label { color: #8b949e !important; }
  [data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #58a6ff !important; font-size: 1.8rem !important;
  }

  /* KPI anomaly card */
  .kpi-card {
    background: linear-gradient(135deg, #2d1515, #1a0a0a);
    border: 1px solid #ff454577;
    border-radius: 12px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.8rem;
    box-shadow: 0 0 12px rgba(255,69,58,0.2);
  }
  .kpi-card h4 { color: #ff6b6b; margin: 0; }

  /* Chat messages */
  .user-bubble {
    background: #1f6feb33;
    border-radius: 12px 12px 2px 12px;
    padding: 0.7rem 1rem;
    margin: 0.4rem 0;
    border: 1px solid #1f6feb55;
  }
  .assistant-bubble {
    background: #21262d;
    border-radius: 12px 12px 12px 2px;
    padding: 0.7rem 1rem;
    margin: 0.4rem 0;
    border: 1px solid #30363d;
  }

  /* Download button */
  .stDownloadButton > button {
    background: linear-gradient(90deg, #1a7f5a, #239970) !important;
    color: white !important; border-radius: 8px !important;
  }

  /* Primary buttons */
  .stButton > button {
    background: linear-gradient(90deg, #1f6feb, #388bfd) !important;
    color: white !important; border: none !important; border-radius: 8px !important;
  }

  /* Form */
  [data-testid="stForm"] {
    background: #161b22; border: 1px solid #30363d;
    border-radius: 12px; padding: 1.5rem;
  }

  /* Tabs header */
  .tab-header {
    font-size: 1.5rem; font-weight: 700;
    background: linear-gradient(90deg, #58a6ff, #79c0ff);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin-bottom: 0.3rem;
  }
  .divider { border-top: 1px solid #21262d; margin: 1rem 0; }

  /* Scrollable chat box */
  .chat-scroll { max-height: 420px; overflow-y: auto; padding-right: 4px; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Helper: API calls with error handling
# ─────────────────────────────────────────────

@st.cache_data(ttl=30, show_spinner=False)
def fetch_all_readings(anomaly_only: bool = False) -> pd.DataFrame:
    try:
        params = {"anomaly_only": "true" if anomaly_only else "false", "limit": 5000}
        r = requests.get(f"{API_BASE}/readings", params=params, timeout=10)
        r.raise_for_status()
        return pd.DataFrame(r.json())
    except Exception as e:
        st.warning(f"⚠️ Could not load readings: {e}")
        return pd.DataFrame()





@st.cache_data(ttl=30, show_spinner=False)
def fetch_sensor_readings(sensor_id: str) -> pd.DataFrame:
    try:
        r = requests.get(f"{API_BASE}/readings/{sensor_id}", timeout=10)
        r.raise_for_status()
        return pd.DataFrame(r.json())
    except Exception as e:
        st.warning(f"⚠️ Could not load readings for {sensor_id}: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=60, show_spinner=False)
def fetch_sensors() -> pd.DataFrame:
    try:
        r = requests.get(f"{API_BASE}/sensors", timeout=10)
        r.raise_for_status()
        return pd.DataFrame(r.json())
    except Exception as e:
        st.warning(f"⚠️ Could not load sensor list: {e}")
        return pd.DataFrame()


def pm25_color(val: float) -> str:
    """Return a CSS color string for a given PM2.5 value."""
    if val is None:   return "#8b949e"
    if val < 12:      return "#3fb950"   # Good — green
    if val < 35:      return "#e3b341"   # Moderate — yellow
    if val < 55:      return "#f0883e"   # Unhealthy for sensitive — orange
    if val < 150:     return "#ff7b72"   # Unhealthy — red
    return "#ff453a"                     # Hazardous — deep red


def pm25_label(val: float) -> str:
    if val is None:   return "Unknown"
    if val < 12:      return "Good"
    if val < 35:      return "Moderate"
    if val < 55:      return "Unhealthy for Sensitive"
    if val < 150:     return "Unhealthy"
    return "Very Unhealthy / Hazardous"


# ─────────────────────────────────────────────
# Session State Initialization
# ─────────────────────────────────────────────
if "ui_target" not in st.session_state:
    st.session_state.ui_target = {
        "tab": "🗺️ Dashboard",
        "lat": 28.6139,
        "lon": 77.2090,
        "zoom": 12,
        "sensor_id": None
    }

# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.image(
        "https://img.icons8.com/fluency/96/wind.png",
        width=72,
    )
    st.markdown("## 🌿 Vayu-Rakshak")
    st.markdown("*Hyperlocal Air Quality Monitor*")
    st.markdown("---")

    # Find index of current tab in list for radio default
    tab_list = ["🗺️ Dashboard", "🚨 Surveillance", "➕ Register Sensor", "📈 Historical Analytics"]
    try:
        tab_index = tab_list.index(st.session_state.ui_target["tab"])
    except ValueError:
        tab_index = 0

    page = st.radio(
        "Navigation",
        tab_list,
        index=tab_index,
        label_visibility="collapsed",
        key="nav_radio"
    )
    # Sync session state if user clicks radio manually
    st.session_state.ui_target["tab"] = page

    st.markdown("---")
    st.markdown("**API Server**")
    api_status_placeholder = st.empty()
    try:
        r = requests.get(f"{API_BASE}/health", timeout=3)
        if r.status_code == 200:
            api_status_placeholder.success("🟢 Online")
        else:
            api_status_placeholder.error("🔴 Error")
    except Exception:
        api_status_placeholder.error("🔴 Offline")

    st.markdown(f"<small style='color:#8b949e'>Endpoint: {API_BASE}</small>", unsafe_allow_html=True)
    st.markdown("---")

    # OpenAI key (optional — only needed for chatbot)
    openai_key = st.text_input(
        "OpenAI API Key (for chatbot)",
        type="password",
        placeholder="sk-...",
        help="Required only for the Dr. Vayu chatbot. Leave blank to disable it.",
    )


# ═══════════════════════════════════════════════════════
# TAB 1: DASHBOARD
# ═══════════════════════════════════════════════════════
if page == "🗺️ Dashboard":
    st.markdown('<p class="tab-header">🗺️ City Air Quality Dashboard</p>', unsafe_allow_html=True)
    st.markdown("Real-time PM2.5 heatmap across all sensors. Brighter zones = higher pollution.")
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # Date/time filter
    col_d1, col_d2, col_d3 = st.columns([1, 1, 2])
    with col_d1:
        start_date = st.date_input("Start Date", value=date.today() - timedelta(days=7))
    with col_d2:
        end_date   = st.date_input("End Date",   value=date.today())
    with col_d3:
        pm25_threshold = st.slider(
            "Show readings with PM2.5 corrected ≥",
            min_value=0, max_value=300, value=0, step=5,
        )

    with st.spinner("Loading sensor data…"):
        df = fetch_all_readings()

    if df.empty:
        st.info("No data found. Ingest some sensor readings or check IF the API is running.")
    else:
        # Parse timestamp
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df_filtered = df[
            (df["timestamp"].dt.date >= start_date) &
            (df["timestamp"].dt.date <= end_date) &
            (df["pm2p5_corrected"].fillna(0) >= pm25_threshold)
        ]

        # KPI row
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("📡 Sensors", df_filtered["sensor_id"].nunique())
        k2.metric("📊 Readings",  len(df_filtered))
        avg_pm = df_filtered["pm2p5_corrected"].mean()
        max_pm = df_filtered["pm2p5_corrected"].max()
        k3.metric("📏 Avg PM2.5",  f"{avg_pm:.1f} µg/m³" if not pd.isna(avg_pm) else "N/A")
        k4.metric("⚠️ Max PM2.5", f"{max_pm:.1f} µg/m³" if not pd.isna(max_pm) else "N/A",
                  delta_color="inverse")

        st.markdown("")

        # Folium heatmap -> replaced with 2km radius Circles and Text Labels
        if df_filtered.dropna(subset=["lat", "long", "pm2p5_corrected"]).empty:
            st.warning("No readings with valid coordinates in the selected date range.")
        else:
            latest = df_filtered.sort_values("timestamp").groupby("sensor_id").last().reset_index()
            latest = latest.dropna(subset=["lat", "long", "pm2p5_corrected"])

            if not latest.empty:
                # Centre map on target or mean of sensors
                m = folium.Map(
                    location=[st.session_state.ui_target["lat"], st.session_state.ui_target["lon"]],
                    zoom_start=st.session_state.ui_target["zoom"],
                    tiles="http://mt0.google.com/vt/lyrs=m&hl=en&x={x}&y={y}&z={z}",
                    attr="Google",
                )

                # Draw points for each sensor
                for _, row in latest.iterrows():
                    val = row["pm2p5_corrected"]
                    color = pm25_color(val)
                    # 1km Coverage Circle
                    folium.Circle(
                        location=[row["lat"], row["long"]],
                        radius=1000,
                        color=color,
                        fill=True,
                        fill_color=color,
                        fill_opacity=0.15,
                        weight=1,
                    ).add_to(m)

                    # Sensor Center Dot
                    folium.CircleMarker(
                        location=[row["lat"], row["long"]],
                        radius=8,
                        color=color,
                        fill=True,
                        fill_color=color,
                        fill_opacity=0.85,
                        popup=folium.Popup(
                            f"<b>{row.get('location_name', row['sensor_id'])}</b><br>"
                            f"Sensor: {row['sensor_id']}<br>"
                            f"PM2.5: <b>{val:.1f} µg/m³</b><br>"
                            f"Status: {pm25_label(val)}<br>"
                            f"Last update: {row['timestamp']}",
                            max_width=220,
                        ),
                        tooltip=f"{row['sensor_id']} — {val:.1f} µg/m³",
                    ).add_to(m)



                st_folium(m, use_container_width=True, height=560)

            st.caption(
                "🔵 Good  🟢 Moderate  🟡 Unhealthy for Sensitive  🔴 Unhealthy  ⛔ Very Unhealthy"
            )

# ═══════════════════════════════════════════════════════
# TAB 2: SURVEILLANCE
# ═══════════════════════════════════════════════════════
elif page == "🚨 Surveillance":
    st.markdown('<p class="tab-header">🚨 Anomaly & Failure Surveillance</p>', unsafe_allow_html=True)
    st.markdown("Sensors flagged with `is_anomaly = 1` or `is_failure = 1` are shown below.")
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    with st.spinner("Loading anomaly data…"):
        df_all   = fetch_all_readings(anomaly_only=False)

    if df_all.empty:
        st.info("No data available.")
    else:
        df_alert = df_all[(df_all["is_anomaly"] == 1) | (df_all["is_failure"] == 1)]

        # Summary KPIs
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("🔴 Anomaly Readings",  int((df_all["is_anomaly"] == 1).sum()))
        m2.metric("⚙️ Failure Readings",  int((df_all["is_failure"] == 1).sum()))
        m3.metric("📡 Affected Sensors",  df_alert["sensor_id"].nunique())
        m4.metric("🔆 Max PM2.5 in Alerts",
                  f"{df_alert['pm2p5_corrected'].max():.1f} µg/m³"
                  if not df_alert.empty else "N/A")

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

        if df_alert.empty:
            st.success("✅ No anomalies or sensor failures detected in the current dataset.")
        else:
            # Per-sensor KPI cards
            st.subheader("Flagged Sensors")
            sensors_flagged = df_alert.groupby("sensor_id").agg(
                location_name=("location_name", "first"),
                lat=("lat", "first"),
                long=("long", "first"),
                anomaly_count=("is_anomaly", "sum"),
                failure_count=("is_failure", "sum"),
                max_pm25=("pm2p5_corrected", "max"),
                last_seen=("timestamp", "max"),
            ).reset_index()

            for _, row in sensors_flagged.iterrows():
                flags = []
                if row["anomaly_count"] > 0:
                    flags.append(f"🔴 {int(row['anomaly_count'])} Anomalies")
                if row["failure_count"] > 0:
                    flags.append(f"⚙️ {int(row['failure_count'])} Failures")
                flag_str = "  |  ".join(flags)

                st.markdown(
                    f"""<div class="kpi-card">
                        <h4>📡 {row['sensor_id']} — {row.get('location_name', 'Unknown')}</h4>
                        <p style="margin:4px 0; color:#ccc;">
                          {flag_str} &nbsp;|&nbsp; Max PM2.5: <b>{row['max_pm25']:.1f} µg/m³</b>
                          &nbsp;|&nbsp; Last seen: {row['last_seen']}
                        </p>
                        <p style="margin:0; color:#8b949e; font-size:0.85rem;">
                          📍 ({row['lat']:.4f}, {row['long']:.4f})
                        </p>
                      </div>""",
                    unsafe_allow_html=True,
                )

            # Full alert table
            st.markdown("---")
            st.subheader("Detailed Alert Log")
            disp = df_alert[[
                "timestamp", "sensor_id", "location_name", "pm2p5_raw",
                "pm2p5_corrected", "temperature", "humidity", "is_anomaly", "is_failure"
            ]].sort_values("timestamp", ascending=False)

            st.dataframe(
                disp.style.applymap(
                    lambda v: "background-color: #3d1c1c;" if v == 1 else "",
                    subset=["is_anomaly", "is_failure"],
                ),
                use_container_width=True,
                height=400,
            )


# ═══════════════════════════════════════════════════════
# TAB 3: REGISTER SENSOR
# ═══════════════════════════════════════════════════════
elif page == "➕ Register Sensor":
    st.markdown('<p class="tab-header">➕ Register a New Sensor</p>', unsafe_allow_html=True)
    st.markdown(
        "Add a new IoT sensor to the Vayu-Rakshak network. "
        "You will receive an **API key** — keep it safe; it is required to authenticate data ingestion."
    )
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    col_form, col_info = st.columns([3, 2])

    with col_form:
        with st.form("register_sensor_form", clear_on_submit=True):
            st.markdown("#### Sensor Details")
            sensor_id_input      = st.text_input("Sensor ID *", placeholder="e.g. ARI-2100")
            location_name_input  = st.text_input("Location Name *", placeholder="e.g. Connaught Place, Delhi")
            c1, c2 = st.columns(2)
            with c1:
                lat_input  = st.number_input("Latitude *",  value=28.6139, format="%.6f", step=0.0001)
            with c2:
                long_input = st.number_input("Longitude *", value=77.2090, format="%.6f", step=0.0001)
            install_date_input = st.date_input("Installation Date", value=date.today())

            submitted = st.form_submit_button("🚀 Register Sensor", use_container_width=True)

        if submitted:
            if not sensor_id_input.strip():
                st.error("Sensor ID is required.")
            elif not location_name_input.strip():
                st.error("Location Name is required.")
            else:
                payload = {
                    "sensor_id":          sensor_id_input.strip(),
                    "location_name":      location_name_input.strip(),
                    "lat":                lat_input,
                    "long":               long_input,
                    "installation_date":  str(install_date_input),
                }
                with st.spinner("Registering sensor…"):
                    try:
                        r = requests.post(
                            f"{API_BASE}/register_sensor",
                            json=payload,
                            timeout=10,
                        )
                        if r.status_code == 201:
                            data = r.json()
                            st.success(f"✅ Sensor **{data['sensor_id']}** registered successfully!")
                            st.code(
                                f"API Key: {data['api_key']}\n\n"
                                "⚠️ Store this key securely. It cannot be recovered later.\n"
                                "Use it as the 'x-api-key' header in /ingest requests.",
                                language="text"
                            )
                            # Invalidate sensors cache
                            fetch_sensors.clear()
                        elif r.status_code == 409:
                            st.warning(r.json().get("detail", "Sensor already exists."))
                        else:
                            st.error(f"Error {r.status_code}: {r.json().get('detail', 'Unknown error')}")
                    except requests.exceptions.ConnectionError:
                        st.error(f"Cannot connect to API at {API_BASE}. Is the FastAPI server running?")
                    except Exception as e:
                        st.error(f"Unexpected error: {e}")

    with col_info:
        st.markdown("#### 🔑 How API Key Works")
        st.info(
            "After registration, you receive a **UUID v4 API key**.\n\n"
            "Every `/ingest` request from this sensor **must include** the header:\n"
            "```\nx-api-key: <your-api-key>\n```\n"
            "Requests without a valid key are rejected with `403 Forbidden`."
        )
        st.markdown("#### 📍 Finding Coordinates")
        st.markdown(
            "[Google Maps](https://maps.google.com) → right-click on a location "
            "→ *What's here?* → copy lat/lng."
        )

        # Live preview map
        st.markdown("#### Live Location Preview")
        preview_map = folium.Map(location=[lat_input, long_input], zoom_start=14,
                                 tiles="http://mt0.google.com/vt/lyrs=m&hl=en&x={x}&y={y}&z={z}",
                                 attr="Google")
        folium.Marker(
            [lat_input, long_input],
            popup=location_name_input or "New Sensor",
            icon=folium.Icon(color="green", icon="cloud"),
        ).add_to(preview_map)
        st_folium(preview_map, height=280, use_container_width=True, key="register_map")


# ═══════════════════════════════════════════════════════
# TAB 4: HISTORICAL ANALYTICS
# ═══════════════════════════════════════════════════════
elif page == "📈 Historical Analytics":
    st.markdown('<p class="tab-header">📈 Historical Analytics</p>', unsafe_allow_html=True)
    st.markdown("Explore PM2.5 trends over time for any sensor and export data as CSV.")
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    sensors_df = fetch_sensors()

    if sensors_df.empty:
        st.info("No sensors registered. Go to **Register Sensor** to add one.")
    else:
        sensor_ids = sensors_df["sensor_id"].tolist()

        col_sel, col_export = st.columns([3, 1])
        with col_sel:
            # Determine default index based on ui_target
            default_sensor_idx = 0
            if st.session_state.ui_target["sensor_id"] in sensor_ids:
                default_sensor_idx = sensor_ids.index(st.session_state.ui_target["sensor_id"])
            
            selected_sensor = st.selectbox("Select Sensor", sensor_ids, index=default_sensor_idx)
            # Sync back if user changes manually
            st.session_state.ui_target["sensor_id"] = selected_sensor

        with st.spinner(f"Loading data for {selected_sensor}…"):
            sensor_df = fetch_sensor_readings(selected_sensor)

        with col_export:
            if not sensor_df.empty:
                csv_bytes = sensor_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="⬇️ Export CSV",
                    data=csv_bytes,
                    file_name=f"{selected_sensor}_readings.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

        if sensor_df.empty:
            st.info(f"No readings found for sensor **{selected_sensor}**.")
        else:
            sensor_df["timestamp"] = pd.to_datetime(sensor_df["timestamp"], errors="coerce")
            sensor_df = sensor_df.sort_values("timestamp")

            # KPI row
            k1, k2, k3, k4 = st.columns(4)
            latest_row = sensor_df.iloc[-1]
            k1.metric("Last PM2.5 (corrected)",
                      f"{latest_row['pm2p5_corrected']:.1f} µg/m³"
                      if pd.notna(latest_row['pm2p5_corrected']) else "N/A")
            k2.metric("Last PM2.5 (raw)",
                      f"{latest_row['pm2p5_raw']:.1f} µg/m³"
                      if pd.notna(latest_row['pm2p5_raw']) else "N/A")
            k3.metric("Total Readings", len(sensor_df))
            k4.metric("Anomalies Flagged", int(sensor_df["is_anomaly"].sum()))

            # Plotly dual-line chart
            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=sensor_df["timestamp"],
                y=sensor_df["pm2p5_raw"],
                name="PM2.5 Raw",
                line=dict(color="#8b949e", width=1.5, dash="dot"),
                opacity=0.7,
            ))

            fig.add_trace(go.Scatter(
                x=sensor_df["timestamp"],
                y=sensor_df["pm2p5_corrected"],
                name="PM2.5 Corrected (AI)",
                line=dict(color="#58a6ff", width=2.5),
                fill="tozeroy",
                fillcolor="rgba(88,166,255,0.07)",
            ))

            # WHO threshold line
            fig.add_hline(
                y=150, line_dash="dash", line_color="#ff7b72", line_width=1,
                annotation_text="WHO Hazardous (150 µg/m³)",
                annotation_position="top left",
                annotation_font_color="#ff7b72",
            )

            fig.update_layout(
                title=dict(
                    text=f"PM2.5 Timeline — Sensor {selected_sensor}",
                    font=dict(size=18, color="#e6edf3"),
                ),
                xaxis=dict(
                    title="Timestamp", showgrid=True, gridcolor="#21262d",
                    color="#8b949e",
                ),
                yaxis=dict(
                    title="PM2.5 (µg/m³)", showgrid=True, gridcolor="#21262d",
                    color="#8b949e",
                ),
                legend=dict(
                    bgcolor="rgba(0,0,0,0)", font=dict(color="#e6edf3", size=12),
                ),
                paper_bgcolor="#0d1117",
                plot_bgcolor="#0d1117",
                hovermode="x unified",
                height=450,
                margin=dict(t=60, b=40, l=60, r=30),
            )

            st.plotly_chart(fig, use_container_width=True)

            # Expandable raw data table
            with st.expander("📋 View Raw Data Table"):
                st.dataframe(
                    sensor_df[[
                        "timestamp", "pm2p5_raw", "pm2p5_corrected",
                        "temperature", "humidity", "is_anomaly", "is_failure"
                    ]].sort_values("timestamp", ascending=False),
                    use_container_width=True,
                )


# ═══════════════════════════════════════════════════════
# PERSISTENT CHATBOT — Dr. Vayu
# ═══════════════════════════════════════════════════════
st.markdown("---")
st.markdown("### 🤖 Dr. Vayu — AI Environmental Scientist")
st.markdown(
    "Ask me anything about the air quality data. "
    "I can query the database, find nearby pollution sources, and explain sensor readings."
)

if not openai_key:
    st.info(
        "💡 Enter your **OpenAI API Key** in the sidebar to activate Dr. Vayu. "
        "The chatbot uses GPT-4o-mini with a LangChain SQL agent."
    )
else:
    # Initialise session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "agent_executor" not in st.session_state:
        st.session_state.agent_executor = None

    # Lazy-load agent
    if st.session_state.agent_executor is None:
        with st.spinner("Initialising Dr. Vayu agent…"):
            try:
                from agent import get_agent_executor
                st.session_state.agent_executor = get_agent_executor(openai_key)
                st.success("✅ Dr. Vayu is ready!")
            except Exception as e:
                st.error(f"Could not initialise agent: {e}")

    # Render chat history
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.chat_history:
            role = msg["role"]
            content = msg["content"]
            if role == "user":
                with st.chat_message("user", avatar="🧑‍💻"):
                    st.markdown(content)
            else:
                with st.chat_message("assistant", avatar="🌿"):
                    st.markdown(content)

    # Input
    user_input = st.chat_input(
        "e.g. 'Why is PM2.5 high at ARI-1885 in the morning?' or 'Show me all anomalies today.'"
    )

    if user_input and st.session_state.agent_executor:
        # Add user message immediately
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user", avatar="🧑‍💻"):
            st.markdown(user_input)

        # Run agent
        with st.chat_message("assistant", avatar="🌿"):
            with st.spinner("Dr. Vayu is analysing data and nearby sources…"):
                try:
                    response = st.session_state.agent_executor.invoke({"input": user_input})
                    raw_answer = response.get("output", "I could not find a relevant answer.")
                    
                    # Intercept UI signals
                    match = re.search(r"UI_SIGNAL: action=(\w+), params=(.+)", raw_answer)
                    if match:
                        action = match.group(1)
                        params = json.loads(match.group(2))
                        
                        if action == "navigate_to":
                            st.session_state.ui_target["tab"] = params.get("tab", st.session_state.ui_target["tab"])
                            if "sensor_id" in params:
                                st.session_state.ui_target["sensor_id"] = params["sensor_id"]
                        elif action == "zoom_to":
                            st.session_state.ui_target["tab"] = "🗺️ Dashboard"
                            st.session_state.ui_target["lat"] = params.get("lat", st.session_state.ui_target["lat"])
                            st.session_state.ui_target["lon"] = params.get("lon", st.session_state.ui_target["lon"])
                            st.session_state.ui_target["zoom"] = params.get("zoom", 14)
                        
                        # Strip the raw signal from the displayed message
                        answer = raw_answer.split("UI_SIGNAL:")[0].strip()
                        if not answer:
                            answer = f"Sure! I've updated the UI for you."
                        st.rerun() # Trigger rerun to reflect UI changes
                    else:
                        answer = raw_answer
                except Exception as e:
                    answer = f"⚠️ Error during analysis: {e}"
            st.markdown(answer)
            st.session_state.chat_history.append({"role": "assistant", "content": answer})

    # Clear chat button
    if st.session_state.chat_history:
        if st.button("🗑️ Clear Conversation", key="clear_chat"):
            st.session_state.chat_history = []
            st.rerun()
