"""
Streamlit Web Dashboard for LumiTrack (Variant 10.03: Museum Exposure Limiter)

Designed with a high-contrast pure white theme (#FFFFFF), modular tabbed navigation,
and crystal-clear visual hierarchy for second-year CSE students, professors, and evaluators.
"""

import sys
import os
import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# Setup import paths
APP_DIR = os.path.dirname(os.path.abspath(__file__))
ML_DIR = os.path.abspath(os.path.join(APP_DIR, "..", "02_Machine_Learning"))
ML_SRC_DIR = os.path.join(ML_DIR, "src")
ML_MODELS_DIR = os.path.join(ML_DIR, "models")

sys.path.append(APP_DIR)
sys.path.append(ML_SRC_DIR)

from host_controller import LumiTrackHostController
from feature_extraction import APPROVED_ANGLES_DEG, FEATURE_NAMES

# Streamlit Page Setup
st.set_page_config(
    page_title="LumiTrack: Museum Light Limiter",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-Contrast Pure-White CSS
st.markdown("""
<style>
    /* Force Pure White Background Across Whole Page */
    .stApp {
        background-color: #FFFFFF !important;
        color: #0F172A !important;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    }
    
    /* Clean Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #F8FAFC !important;
        border-right: 1px solid #E2E8F0 !important;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #F1F5F9;
        padding: 6px;
        border-radius: 12px;
        border: 1px solid #E2E8F0;
    }

    .stTabs [data-baseweb="tab"] {
        height: 44px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 8px;
        color: #475569;
        font-weight: 600;
        padding: 0 18px;
        border: none;
    }

    .stTabs [aria-selected="true"] {
        background-color: #FFFFFF !important;
        color: #2563EB !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08) !important;
    }

    /* Streamlit Metric Overrides */
    div[data-testid="stMetric"] {
        background-color: #FFFFFF !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 12px !important;
        padding: 14px 18px !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04) !important;
    }

    div[data-testid="stMetricLabel"] {
        color: #64748B !important;
        font-size: 0.85rem !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.04em !important;
    }

    div[data-testid="stMetricValue"] {
        color: #0F172A !important;
        font-weight: 800 !important;
        font-size: 1.8rem !important;
    }

    /* Custom Alert Boxes */
    .alert-danger-box {
        background-color: #FEF2F2;
        border: 1px solid #FCA5A5;
        border-left: 6px solid #DC2626;
        border-radius: 10px;
        padding: 18px 22px;
        margin-bottom: 20px;
    }

    .alert-safe-box {
        background-color: #ECFDF5;
        border: 1px solid #6EE7B7;
        border-left: 6px solid #059669;
        border-radius: 10px;
        padding: 18px 22px;
        margin-bottom: 20px;
    }

    .alert-confuser-box {
        background-color: #FFFBEB;
        border: 1px solid #FCD34D;
        border-left: 6px solid #D97706;
        border-radius: 10px;
        padding: 18px 22px;
        margin-bottom: 20px;
    }

    /* Section Cards */
    .content-box {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.03);
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# 13 Features metadata for educational clarity
FEATURE_METADATA = {
    "peak_intensity": {
        "unit": "Lux",
        "category": "1. Intensity Extremes",
        "desc": "Highest light level recorded anywhere during the 7-angle scan.",
        "why": "Direct indicator of hazardous spotlight or sunlight intrusion."
    },
    "peak_angle_deg": {
        "unit": "Degrees (°)",
        "category": "3. Geometry & Direction",
        "desc": "Exact servo angle (15° to 165°) where the peak light originated.",
        "why": "Enables the actuator to steer louvers away from the glare source."
    },
    "min_intensity": {
        "unit": "Lux",
        "category": "1. Intensity Extremes",
        "desc": "Lowest light level recorded across the 7 scan positions.",
        "why": "Establishes the ambient dark room floor inside the gallery."
    },
    "mean_intensity": {
        "unit": "Lux",
        "category": "1. Intensity Extremes",
        "desc": "Arithmetic average light across all 7 points.",
        "why": "Measures total broad illumination across the exhibit envelope."
    },
    "std_intensity": {
        "unit": "Lux",
        "category": "2. Spatial Heterogeneity",
        "desc": "Standard deviation measuring how concentrated the beam is.",
        "why": "High std indicates sharp beam/glint; low std indicates soft daylight."
    },
    "curve_asymmetry": {
        "unit": "Ratio [-1, 1]",
        "category": "3. Geometry & Direction",
        "desc": "Balance between right wing (115-165°) and left wing (15-65°).",
        "why": "Identifies if sunlight is streaming from the left (-1) or right (+1) window."
    },
    "gradient_mean": {
        "unit": "Lux/deg",
        "category": "2. Spatial Heterogeneity",
        "desc": "Average rate of light change between adjacent servo steps.",
        "why": "Measures overall spatial roughness of the light field."
    },
    "gradient_max": {
        "unit": "Lux/deg",
        "category": "2. Spatial Heterogeneity",
        "desc": "Sharpest single jump in light between two neighboring angles.",
        "why": "Extremely high for glass glints; distinguishes reflections from true beams."
    },
    "contrast_ratio": {
        "unit": "Ratio [0, 1]",
        "category": "2. Spatial Heterogeneity",
        "desc": "Michelson contrast: (Peak - Min) / (Peak + Min).",
        "why": "1.0 represents localized harsh glare; 0.2 represents gentle ambient light."
    },
    "exposure_integral": {
        "unit": "Lux·deg",
        "category": "4. Conservation Flux & Doses",
        "desc": "Trapezoidal numerical flux integral across the entire angular sweep.",
        "why": "Quantifies total photochemical radiant energy depositing on the artwork."
    },
    "ratio_to_target": {
        "unit": "Ratio",
        "category": "4. Conservation Flux & Doses",
        "desc": "Ratio of peak intensity to operator potentiometer limit.",
        "why": "Directly benchmarks observed light against curator safety guidelines."
    },
    "ratio_to_ambient": {
        "unit": "Ratio",
        "category": "4. Conservation Flux & Doses",
        "desc": "Ratio of peak intensity to dark reference baseline.",
        "why": "Measures beam intensity multiplier over background gallery darkness."
    },
    "partial_slope": {
        "unit": "Lux/deg",
        "category": "3. Geometry & Direction",
        "desc": "Initial spatial slope from 15° to neutral home (90°).",
        "why": "Enables rapid early detection of morning solar drift."
    }
}


@st.cache_data
def load_metrics_summary():
    """Loads precomputed model comparison and evaluation metrics."""
    summary_path = os.path.join(ML_MODELS_DIR, "metrics_summary.json")
    if os.path.exists(summary_path):
        with open(summary_path, "r") as f:
            return json.load(f)
    return None


metrics_summary = load_metrics_summary()

# Initialize session state variables
if "controller" not in st.session_state:
    st.session_state.controller = None
if "connected" not in st.session_state:
    st.session_state.connected = False
if "history" not in st.session_state:
    st.session_state.history = []
if "conn_error" not in st.session_state:
    st.session_state.conn_error = None

# ==============================================================================
# SIDEBAR CONTROLS
# ==============================================================================
with st.sidebar:
    st.header("⚙️ 1. Hardware Connection")
    mode = st.radio(
        "Operating Mode:",
        ["Simulation (Virtual STM32)", "Live Hardware (ST-LINK USB)"],
        help="Simulation runs without physical board. Live Hardware communicates with STM32 NUCLEO-F446RE over UART."
    )
    is_live = (mode == "Live Hardware (ST-LINK USB)")

    if is_live:
        try:
            import serial.tools.list_ports
            detected_ports = list(serial.tools.list_ports.comports())
        except Exception:
            detected_ports = []

        if detected_ports:
            port_names = [p.device for p in detected_ports]
            default_idx = 0
            for idx, p in enumerate(detected_ports):
                if "STLink" in p.description or "STM32" in p.description or p.device == "COM5":
                    default_idx = idx
                    break
            port = st.selectbox(
                "Detected Serial Port:",
                options=port_names,
                index=default_idx,
                format_func=lambda d: f"{d} ({next((p.description for p in detected_ports if p.device == d), '')})",
                help="Automatically detects your STM32 ST-LINK Virtual COM port."
            )
        else:
            port = st.text_input("Serial COM Port:", value="COM5")
        baud = st.selectbox("Baud Rate:", [115200, 9600, 57600], index=0)
        scenario = "OVEREXPOSURE_RISK"
    else:
        port = None
        baud = 115200
        scenario = st.selectbox(
            "Simulation Optical Scenario:",
            ["OVEREXPOSURE_RISK", "NORMAL_DIFFUSE", "CONFUSER_SPECULAR"],
            help="Select which physical lighting condition the virtual STM32 emulator will generate."
        )

    col_conn, col_dis = st.columns(2)
    with col_conn:
        if st.button("🔌 Connect", disabled=st.session_state.connected, use_container_width=True):
            ctrl_mode = "live" if is_live else "sim"
            ctrl = LumiTrackHostController(mode=ctrl_mode, port=port, baudrate=baud, scenario=scenario)
            try:
                ctrl.connect()
                st.session_state.controller = ctrl
                st.session_state.connected = True
                st.session_state.conn_error = None
                st.rerun()
            except Exception as e:
                st.session_state.controller = None
                st.session_state.connected = False
                st.session_state.conn_error = str(e)
                st.rerun()

    with col_dis:
        if st.button("❌ Disconnect", disabled=not st.session_state.connected, use_container_width=True):
            if st.session_state.controller:
                st.session_state.controller.disconnect()
            st.session_state.controller = None
            st.session_state.connected = False
            st.session_state.conn_error = None
            st.rerun()

    if st.session_state.get("conn_error"):
        st.error(f"""
        ⚠️ **Connection Error on `{port}`**:
        `{st.session_state.conn_error}`
        
        **Quick Solution:**
        1. **Unplug & Re-plug USB**: Unplug your STM32 Nucleo USB cable from the PC, wait 3 seconds, and plug it back in. This clears the Windows port lock!
        2. **Close other serial tools**: Make sure STM32CubeIDE serial monitor, PuTTY, or Arduino IDE are closed.
        3. Click **Connect** again!
        """)

    st.divider()

    # Active ML Model Selector
    st.header("🧠 2. Active ML Model")
    best_name = metrics_summary.get("selected_best_model", "Random Forest Classifier") if metrics_summary else "Random Forest Classifier"

    model_options = [
        f"Best Model ({best_name})",
        "Ridge Logistic Regression",
        "Random Forest Classifier",
        "Gradient Boosting Classifier",
        "Rule-Based Baseline"
    ]
    selected_model_option = st.selectbox(
        "Model for Primary Prediction:",
        options=model_options,
        index=0,
        help="Defaults to Random Forest (selected by 5-fold cross-validation). You can switch to any model to test their predictions."
    )

    if "Best Model" in selected_model_option:
        active_model_name = "Best Model"
    else:
        active_model_name = selected_model_option

    st.divider()

    st.header("🕹️ 3. Physical Actuator Controls")
    st.caption("Send commands directly to the STM32 board:")

    c_b1, c_b2 = st.columns(2)
    with c_b1:
        if st.button("🔄 Full Scan", disabled=not st.session_state.connected, use_container_width=True, help="Sweeps 7 angles (15° to 165°) to map the light field and trigger ML."):
            st.session_state.controller.trigger_scan(profile="FULL")
            st.toast("🔄 7-Angle Optical Scan Initiated...")
            st.rerun()
    with c_b2:
        if st.button("⚡ Fast Scan", disabled=not st.session_state.connected, use_container_width=True, help="Fast early sweep (15° to 90°)."):
            st.session_state.controller.trigger_scan(profile="PARTIAL")
            st.toast("⚡ Partial Scan Initiated...")
            st.rerun()

    c_b3, c_b4 = st.columns(2)
    with c_b3:
        if st.button("🏠 Safe Home", disabled=not st.session_state.connected, use_container_width=True, help="Moves servo back to neutral 90° position."):
            st.session_state.controller.return_home()
            st.toast("🏠 Moving Servo to Safe Home (90°)...")
            st.rerun()
    with c_b4:
        if st.button("⏸️ Hold Position", disabled=not st.session_state.connected, use_container_width=True, help="Locks current servo orientation."):
            st.session_state.controller.hold()
            st.toast("⏸️ Servo Position Locked.")
            st.rerun()

    st.caption("Manual Servo Steering:")
    sel_angle_id = st.select_slider(
        "Select Approved Angle:",
        options=list(range(len(APPROVED_ANGLES_DEG))),
        format_func=lambda x: f"POS_{x} ({APPROVED_ANGLES_DEG[x]}°)"
    )
    if st.button("🎯 Move to Angle", disabled=not st.session_state.connected, use_container_width=True):
        st.session_state.controller.set_angle(sel_angle_id)
        st.toast(f"🎯 Commanded Servo -> POS_{sel_angle_id} ({APPROVED_ANGLES_DEG[sel_angle_id]}°)")
        st.rerun()

    st.caption("⚡ Quick Angle Presets:")
    q_cols1 = st.columns(4)
    q_cols2 = st.columns(3)
    presets = [
        (0, "POS_0 (15°)"),
        (1, "POS_1 (40°)"),
        (2, "POS_2 (65°)"),
        (3, "POS_3 (90° Safe Home)"),
        (4, "POS_4 (115°)"),
        (5, "POS_5 (140°)"),
        (6, "POS_6 (165°)")
    ]
    for i in range(4):
        aid, lbl = presets[i]
        with q_cols1[i]:
            if st.button(f"{APPROVED_ANGLES_DEG[aid]}°", key=f"quick_ang_{aid}", disabled=not st.session_state.connected, use_container_width=True):
                st.session_state.controller.set_angle(aid)
                st.toast(f"🎯 Commanded Servo -> {lbl}")
                st.rerun()
    for i in range(4, 7):
        aid, lbl = presets[i]
        with q_cols2[i - 4]:
            if st.button(f"{APPROVED_ANGLES_DEG[aid]}°", key=f"quick_ang_{aid}", disabled=not st.session_state.connected, use_container_width=True):
                st.session_state.controller.set_angle(aid)
                st.toast(f"🎯 Commanded Servo -> {lbl}")
                st.rerun()


# ==============================================================================
# MAIN PAGE HEADER & BRANDING
# ==============================================================================
st.title("🏛️ LumiTrack: Intelligent Museum Light Limiter")
st.caption("Parent Problem 10 | Variant 10.03 | STM32 NUCLEO-F446RE + 4 Machine Learning Classifiers")

# ==============================================================================
# TABBED NAVIGATION INTERFACE
# ==============================================================================

if st.session_state.connected and st.session_state.controller:

    # Auto-refresh fragment (1 second smooth update without full page reload)
    @st.fragment(run_every=1.0)
    def render_dashboard_tabs():
        ctrl = st.session_state.controller
        telem = ctrl.latest_telemetry
        scan_res = ctrl.latest_scan_result
        predictor = ctrl.predictor

        # Top System Status Line (dynamically updated every second)
        pkt_num = telem.get('seq', 0)
        cur_lux = telem.get('lux', 0.0)
        if ctrl.mode == "sim":
            st.markdown(f"""
            <div style="background-color: #EFF6FF; border: 1px solid #BFDBFE; border-radius: 8px; padding: 10px 16px; margin-bottom: 16px; display: flex; justify-content: space-between; align-items: center;">
                <div>🟡 <b>Operating Mode: SIMULATION (Virtual STM32 Emulator)</b> — Scenario: <code>{ctrl.scenario}</code></div>
                <div>Status: <span style="color:#059669; font-weight:bold;">Online (115200 baud)</span> | Packet #{pkt_num} ({cur_lux:.1f} Lux)</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="background-color: #ECFDF5; border: 1px solid #A7F3D0; border-radius: 8px; padding: 10px 16px; margin-bottom: 16px; display: flex; justify-content: space-between; align-items: center;">
                <div>🟢 <b>Operating Mode: LIVE HARDWARE (STM32 NUCLEO-F446RE)</b> — Port: <code>{ctrl.port}</code></div>
                <div>Status: <span style="color:#059669; font-weight:bold;">Receiving Physical Telemetry</span> | Packet #{pkt_num} ({cur_lux:.1f} Lux)</div>
            </div>
            """, unsafe_allow_html=True)

        # History tracking for charts
        if telem:
            st.session_state.history.append({
                "timestamp": time.time(),
                "lux": telem.get("lux", 0.0),
                "target_lux": telem.get("target_lux", 500.0),
                "angle_deg": telem.get("angle_deg", 90),
                "ldr_v": telem.get("ldr_v", 0.0),
                "pot_v": telem.get("pot_v", 0.0)
            })
            if len(st.session_state.history) > 60:
                st.session_state.history.pop(0)

        # Tab bar
        tab_monitor, tab_ml, tab_features, tab_hardware = st.tabs([
            "🏛️ 1. Live Exhibit Monitor & Actions",
            "🧠 2. ML Engine & 4 Algorithms",
            "🔬 3. 13 Physical Features Deep-Dive",
            "🔌 4. Hardware Diagnostics & Registers"
        ])

        # ----------------------------------------------------------------------
        # TAB 1: LIVE EXHIBIT MONITOR & ACTIONS
        # ----------------------------------------------------------------------
        with tab_monitor:
            curr_lux = telem.get("lux", 0.0) if telem else 0.0
            target_lux = telem.get("target_lux", 500.0) if telem else 500.0
            curr_deg = telem.get("angle_deg", 90) if telem else 90
            curr_id = telem.get("angle_id", 3) if telem else 3
            cum_lux = telem.get("cum_lux", 0.0) if telem else 0.0
            pct_dose = min(100.0, (cum_lux / 50000.0) * 100.0)

            # Evaluate latest scan if present
            pred_res = None
            all_models_comp = None
            if scan_res and "data" in scan_res and predictor is not None:
                pts = scan_res["data"]
                lux_vals = [p["lux"] for p in pts]
                scan_target = target_lux
                pred_res = predictor.evaluate_full_scan(
                    lux_vals, target_lux=scan_target, ambient_lux=40.0, model_choice=active_model_name
                )
                all_models_comp = predictor.compare_all_models(
                    lux_vals, target_lux=scan_target, ambient_lux=40.0
                )

            # Prominent Action / Status Card
            if pred_res:
                regime = pred_res["regime_name"]
                conf = pred_res["confidence"]
                peak_lux = pred_res["peak_intensity"]
                action_msg = pred_res["action_description"]

                if regime == "OVEREXPOSURE_RISK":
                    st.markdown(f"""
                    <div class="alert-danger-box">
                        <h3 style="color:#991B1B; margin:0 0 6px 0;">🚨 OVEREXPOSURE RISK DETECTED ({conf*100:.1f}% Confidence)</h3>
                        <p style="color:#7F1D1D; margin:0; font-size:1.05rem;">
                            <b>Action Executed:</b> {action_msg}<br>
                            <b>Hazard Reason:</b> The peak light intensity (<b>{peak_lux:.1f} Lux</b>) severely exceeds the museum threshold (<b>{scan_target:.1f} Lux</b>). The servo automatically deflected the shades to protect delicate artwork pigments from irreversible photo-oxidation.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                elif regime == "CONFUSER_SPECULAR":
                    st.markdown(f"""
                    <div class="alert-confuser-box">
                        <h3 style="color:#92400E; margin:0 0 6px 0;">🔍 SPECULAR CONFUSER (GLASS GLINT) RECOGNIZED ({conf*100:.1f}% Confidence)</h3>
                        <p style="color:#78350F; margin:0; font-size:1.05rem;">
                            <b>Action Executed:</b> {action_msg}<br>
                            <b>Intelligent Suppression:</b> A high local spike (<b>{peak_lux:.1f} Lux</b>) occurred, but Machine Learning analyzed the spatial gradient and confirmed this is only a harmless visitor/glass reflection. False alarm suppressed so the exhibit remains viewable!
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="alert-safe-box">
                        <h3 style="color:#065F46; margin:0 0 6px 0;">✅ SAFE DIFFUSE LIGHTING ({conf*100:.1f}% Confidence)</h3>
                        <p style="color:#064E3B; margin:0; font-size:1.05rem;">
                            <b>Action Executed:</b> {action_msg}<br>
                            <b>Conservation Status:</b> Ambient illumination is soft, uniform, and well within conservation tolerances. The exhibit remains viewable at neutral orientation (90°).
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("ℹ️ **No angular scan run yet!** Click **'🔄 Full Scan'** in the left sidebar to sweep the 7 angles and generate an instant ML classification.")

            # Top 4 Primary Live Metric Cards
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric(
                    label="Instant Light Intensity",
                    value=f"{curr_lux:.1f} Lux",
                    delta="Exceeds Limit" if curr_lux > target_lux else "Safe Level",
                    delta_color="inverse" if curr_lux > target_lux else "normal",
                    help="Real-time light level from the optical sensor. 50-200 Lux is typical museum ambient."
                )
            with m2:
                st.metric(
                    label="Operator Limit Threshold",
                    value=f"{target_lux:.1f} Lux",
                    delta=f"Set via Potentiometer ({telem.get('pot_v', 0.0):.2f}V)",
                    help="Conservation limit configured by the curator via the physical 10k potentiometer knob."
                )
            with m3:
                st.metric(
                    label="Servo Kinematic Angle",
                    value=f"{curr_deg}°",
                    delta=f"POS_{curr_id} (Safe Home)" if curr_id == 3 else f"POS_{curr_id}",
                    help="Current mechanical angle of the sensor. 90° is neutral facing the exhibit."
                )
            with m4:
                st.metric(
                    label="Total Photochemical Dose",
                    value=f"{cum_lux:.0f} Lux·s",
                    delta=f"{pct_dose:.1f}% of daily limit",
                    delta_color="inverse" if pct_dose > 80 else "normal",
                    help="Integrated exposure over time. 50,000 Lux·s is the maximum recommended daily limit."
                )

            # Cumulative Exposure Visual Gauge Bar
            st.caption(f"**Photochemical Exposure Accumulation Meter** ({cum_lux:.0f} / 50,000 Lux·s):")
            st.progress(min(1.0, cum_lux / 50000.0))

            st.write("")

            # Main Charts Row
            g1, g2 = st.columns(2)
            with g1:
                st.subheader("🧭 Polar Light-Field Profile")
                st.caption("Maps how bright the light is at each of the 7 servo angles (15° to 165°):")
                if scan_res and "data" in scan_res:
                    pts = scan_res["data"]
                    degs = [p["deg"] for p in pts]
                    luxs = [p["lux"] for p in pts]

                    fig = plt.figure(figsize=(6, 4))
                    ax = fig.add_subplot(111, polar=True)
                    rads = np.radians(degs)
                    ax.plot(rads, luxs, "o-", linewidth=2.5, color="#2563EB", label="Scanned Profile")
                    ax.fill(rads, luxs, color="#93C5FD", alpha=0.35)
                    ax.set_theta_zero_location("N")
                    ax.set_title("Scanned Angular Light Distribution", va="bottom", fontsize=10, fontweight="bold")
                    ax.grid(True, linestyle="--", alpha=0.5)
                    st.pyplot(fig)
                    plt.close()
                else:
                    st.info("Click **'🔄 Full Scan'** in the sidebar to populate this polar map.")

            with g2:
                st.subheader("📈 Live Light Trend Over Time")
                st.caption("Continuous line chart comparing instant light against curator threshold:")
                if len(st.session_state.history) > 2:
                    hist_df = pd.DataFrame(st.session_state.history)
                    st.line_chart(hist_df[["lux", "target_lux"]], use_container_width=True)
                else:
                    st.caption("Accumulating live telemetry readings...")

        # ----------------------------------------------------------------------
        # TAB 2: ML ENGINE & 4 ALGORITHMS COMPARISON
        # ----------------------------------------------------------------------
        with tab_ml:
            st.header("🧠 Machine Learning Comparison Engine")
            st.caption("Side-by-side performance of all four required algorithms on the active light scan:")

            # 4 Algorithms Table
            if metrics_summary and "comparison_table" in metrics_summary:
                comp_rows = metrics_summary["comparison_table"]
                table_data = []

                for r in comp_rows:
                    m_name = r["model_name"]
                    is_bin = r.get("is_binary_baseline", False)

                    live_pred = "-"
                    live_conf = "-"
                    if all_models_comp and m_name in all_models_comp:
                        live_pred = all_models_comp[m_name]["regime"]
                        live_conf = f"{all_models_comp[m_name]['confidence'] * 100:.1f}%"
                    elif all_models_comp and is_bin:
                        live_pred = all_models_comp.get("Rule-Based Baseline", {}).get("regime", "-")
                        live_conf = "100.0%"

                    table_data.append({
                        "Algorithm Name": m_name,
                        "Type": "Binary Rule (Heuristic)" if is_bin else "Multiclass Machine Learning",
                        "CV Macro-F1 (Train)": f"{metrics_summary.get('cross_validation_scores_macro_f1', {}).get(m_name, 0.0):.4f}" if not is_bin else "N/A",
                        "Held-Out Test Accuracy": f"{r['accuracy'] * 100:.2f}%",
                        "Held-Out Macro-F1": f"{r['macro_f1'] * 100:.2f}%",
                        "Current Scan Prediction": live_pred,
                        "Confidence": live_conf
                    })

                st.dataframe(pd.DataFrame(table_data), use_container_width=True)

            # Two ML Visual Charts
            m_c1, m_c2 = st.columns(2)
            with m_c1:
                st.subheader("🍩 Active Model Probabilities")
                st.caption(f"Confidence distribution across the 3 regimes for {active_model_name}:")
                if pred_res is not None and predictor is not None and predictor.best_model is not None:
                    feat_vals = [pred_res["features"][name] for name in FEATURE_NAMES]
                    X_df = pd.DataFrame([feat_vals], columns=FEATURE_NAMES)

                    if hasattr(predictor.best_model, "predict_proba"):
                        probas = predictor.best_model.predict_proba(X_df)[0]
                        labels = ["Normal Diffuse", "Overexposure Risk", "Specular Confuser"]
                        colors = ["#10B981", "#EF4444", "#F59E0B"]

                        fig, ax = plt.subplots(figsize=(5, 3.5))
                        ax.pie(
                            probas,
                            labels=labels,
                            autopct="%1.1f%%",
                            colors=colors,
                            startangle=90,
                            wedgeprops=dict(width=0.45, edgecolor="white", linewidth=2)
                        )
                        ax.set_title(f"Class Probabilities ({predictor.best_model_name})", fontsize=10, fontweight="bold")
                        st.pyplot(fig)
                        plt.close()
                else:
                    st.caption("Available after performing a scan.")

            with m_c2:
                st.subheader("📊 Model Accuracy Comparison")
                st.caption("Benchmarking on the independent held-out test set (120 samples):")
                if metrics_summary and "comparison_table" in metrics_summary:
                    comp_rows = metrics_summary["comparison_table"]
                    chart_df = pd.DataFrame([
                        {
                            "Algorithm": r["model_name"].replace(" Classifier", "").replace(" (Binary Heuristic)", ""),
                            "Accuracy (%)": r["accuracy"] * 100,
                            "Macro-F1 (%)": r["macro_f1"] * 100
                        }
                        for r in comp_rows
                    ]).set_index("Algorithm")
                    st.bar_chart(chart_df, use_container_width=True)

            # Explanatory Educational Notes
            st.markdown("""
            ### 💡 Core Machine Learning Concepts in LumiTrack:
            1. **Why Random Forest was Selected as Best Model**:
               - Evaluated via **5-Fold Stratified Cross-Validation strictly on the training set** to prevent data leakage.
               - Achieved the highest validation Macro-F1 (**0.9930** vs 0.9860 for Ridge LR and 0.9835 for Gradient Boosting).
               - Being an ensemble of 100 decision trees, it handles non-linear optical reflections without needing feature scaling.
            2. **Why the Rule-Based Baseline Fails (26.0% False Alarms)**:
               - The Rule-Based Baseline is a simple binary heuristic (`IF peak > 400 Lux THEN Overexposure`).
               - When a glass showcase or wristwatch reflects a localized 520 Lux glint, the baseline panics and triggers a blind closure!
               - The Machine Learning models examine the entire 13-feature spatial distribution, recognizing high gradients but low total energy, and suppress the false alarm.
            """)

        # ----------------------------------------------------------------------
        # TAB 3: 13 PHYSICAL FEATURES DEEP-DIVE
        # ----------------------------------------------------------------------
        with tab_features:
            st.header("🔬 The 13 Physics-Informed Extracted Features")
            st.caption("Every model receives these exact 13 features in fixed order. See live values and why each feature is critical:")

            if pred_res and "features" in pred_res:
                feat_dict = pred_res["features"]
                feature_rows = []
                for i, name in enumerate(FEATURE_NAMES, start=1):
                    val = feat_dict.get(name, 0.0)
                    meta = FEATURE_METADATA.get(name, {"unit": "-", "category": "-", "desc": "-", "why": "-"})
                    feature_rows.append({
                        "#": i,
                        "Feature Identifier": name,
                        "Current Measured Value": f"{val:.4f}" if isinstance(val, float) else str(val),
                        "Unit": meta["unit"],
                        "Category": meta["category"],
                        "Physical Meaning": meta["desc"],
                        "Conservation Purpose": meta["why"]
                    })
                st.dataframe(pd.DataFrame(feature_rows).set_index("#"), use_container_width=True)
            else:
                st.info("Trigger a **'🔄 Full Scan'** in the sidebar to populate live measured values for all 13 features.")

            st.write("")
            st.subheader("📚 Feature Category Breakdown")
            fc1, fc2, fc3, fc4 = st.columns(4)
            with fc1:
                st.markdown("**1. Intensity Extremes**")
                st.caption("`peak_intensity`, `min_intensity`, `mean_intensity` — Track absolute optical brightness across the gallery.")
            with fc2:
                st.markdown("**2. Spatial Heterogeneity**")
                st.caption("`std_intensity`, `gradient_mean`, `gradient_max`, `contrast_ratio` — Detect beam sharp edges and specular glint spikes.")
            with fc3:
                st.markdown("**3. Geometry & Direction**")
                st.caption("`peak_angle_deg`, `curve_asymmetry`, `partial_slope` — Identify incoming light direction to steer louvers away.")
            with fc4:
                st.markdown("**4. Flux & Conservation Ratios**")
                st.caption("`exposure_integral`, `ratio_to_target`, `ratio_to_ambient` — Total photochemical radiant dose impacting artifact pigments.")

        # ----------------------------------------------------------------------
        # TAB 4: HARDWARE DIAGNOSTICS & REGISTERS
        # ----------------------------------------------------------------------
        with tab_hardware:
            st.header("🔌 STM32 NUCLEO-F446RE Hardware Telemetry")
            st.caption("Real-time electrical registers, voltages, ADC samples, and UART packet inspection:")

            h_c1, h_c2, h_c3, h_c4 = st.columns(4)
            with h_c1:
                st.markdown("**LDR Sensor Channel**")
                st.write(f"• Voltage: **{telem.get('ldr_v', 0.0):.3f} V**")
                st.write(f"• 12-bit Raw ADC: **{telem.get('ldr_raw', 0)}** / 4095")
                st.write(f"• Calibrated Lux: **{telem.get('lux', 0.0):.1f} Lux**")
            with h_c2:
                st.markdown("**10k Potentiometer Channel**")
                st.write(f"• Voltage: **{telem.get('pot_v', 0.0):.3f} V**")
                st.write(f"• 12-bit Raw ADC: **{telem.get('pot_raw', 0)}** / 4095")
                st.write(f"• Target Threshold: **{telem.get('target_lux', 500.0):.1f} Lux**")
            with h_c3:
                st.markdown("**Servo Kinematics & FSM**")
                st.write(f"• FSM State: **{telem.get('state', 'SAFE_HOME')}**")
                st.write(f"• Angle: **{telem.get('angle_deg', 90)}°** (ID: {telem.get('angle_id', 3)})")
                st.write(f"• PWM Frequency: **50 Hz (20 ms period)**")
            with h_c4:
                st.markdown("**Safety Watchdog & Protocol**")
                st.write(f"• Safety Flag: **{'1 (NORMAL)' if telem.get('safe', 1) else '0 (TRIPPED)'}**")
                st.write(f"• Manual Override: **{'1 (ACTIVE)' if telem.get('ovr', 0) else '0 (INACTIVE)'}**")
                st.write(f"• Watchdog Timeout: **3000 ms failsafe**")

            st.write("")
            with st.expander("🔍 Raw JSON Telemetry Packet (Exact UART Payload)", expanded=True):
                st.json(telem)

    # Render dashboard
    render_dashboard_tabs()

else:
    # Disconnected Welcome Guide
    st.markdown("""
    <div style="background-color: #F8FAFC; border: 2px dashed #CBD5E1; border-radius: 12px; padding: 40px; text-align: center; margin-top: 30px;">
        <h2 style="color: #1E293B; margin-bottom: 10px;">🔌 Connect to Start Monitoring</h2>
        <p style="color: #64748B; font-size: 1.1rem; max-width: 650px; margin: 0 auto 20px auto;">
            The dashboard is ready. Follow the simple steps in the left sidebar to connect:
        </p>
        <div style="display: flex; justify-content: center; gap: 30px; text-align: left; max-width: 750px; margin: 0 auto;">
            <div style="background: white; border: 1px solid #E2E8F0; padding: 20px; border-radius: 8px; flex: 1;">
                <h4 style="color: #2563EB; margin-top: 0;">Option A: Simulation Mode</h4>
                <p style="color: #475569; font-size: 0.9rem;">
                    Test the complete system instantly without hardware. Select a scenario (e.g. <code>OVEREXPOSURE_RISK</code>) in the sidebar and click <b>Connect</b>.
                </p>
            </div>
            <div style="background: white; border: 1px solid #E2E8F0; padding: 20px; border-radius: 8px; flex: 1;">
                <h4 style="color: #10B981; margin-top: 0;">Option B: Physical Hardware</h4>
                <p style="color: #475569; font-size: 0.9rem;">
                    Plug your STM32 NUCLEO-F446RE board into USB, select <b>Live Hardware</b> in the sidebar, verify the COM port, and click <b>Connect</b>.
                </p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
