import math
import time
import cv2
import numpy as np
import onnxruntime as ort
import streamlit as st
from PIL import Image
from google import genai

st.set_page_config(
    page_title="SONAR-AI | Maritime Inspection Console",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Custom CSS: Maritime Command Console ---

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

    /* === PALETTE VARIABLES (Currently: Option 1 - Slate & Cobalt Dark) === */
    :root {
        --bg-card: #161B22;
        --bg-subtle: #1C2128;
        --border: #30363D;
        --text-main: #E6EDF3;
        --text-muted: #8B949E;
        --accent: #2F81F7;
        --accent-hover: #1F6FEB;
        --accent-soft: rgba(47, 129, 247, 0.12);
        --status-dot: #3FB950;
        --btn-text: #FFFFFF;
    }

    /* To use Option 2 (Research Lab Light), replace the :root block above with:
    :root {
        --bg-card: #FFFFFF;
        --bg-subtle: #F1F5F9;
        --border: #E2E8F0;
        --text-main: #0F172A;
        --text-muted: #64748B;
        --accent: #1D4ED8;
        --accent-hover: #1E40AF;
        --accent-soft: rgba(29, 78, 216, 0.08);
        --status-dot: #16A34A;
        --btn-text: #FFFFFF;
    }
    */

    /* To use Option 3 (Graphite & Amber Dark), replace the :root block above with:
    :root {
        --bg-card: #1A1D21;
        --bg-subtle: #22262B;
        --border: #2C3036;
        --text-main: #F3F4F6;
        --text-muted: #9CA3AF;
        --accent: #D97706;
        --accent-hover: #B45309;
        --accent-soft: rgba(217, 119, 6, 0.14);
        --status-dot: #10B981;
        --btn-text: #FFFFFF;
    }
    */

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Header Card */
    .hero-banner {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 1.25rem 1.6rem;
        margin-bottom: 1.2rem;
    }
    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 7px;
        background: var(--accent-soft);
        border: 1px solid var(--border);
        color: var(--accent);
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.7rem;
        font-weight: 500;
        letter-spacing: 0.06em;
        padding: 3px 10px;
        border-radius: 6px;
        margin-bottom: 0.55rem;
    }
    .pulse-dot {
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background: var(--status-dot);
        display: inline-block;
    }
    .hero-title {
        font-size: 1.65rem;
        font-weight: 600;
        margin: 0;
        color: var(--text-main);
        letter-spacing: -0.02em;
    }
    .hero-sub {
        color: var(--text-muted);
        font-size: 0.92rem;
        margin-top: 0.3rem;
        margin-bottom: 0;
    }

    /* Clean Telemetry Cards */
    .hud-card {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 0.95rem 1.15rem;
    }
    .hud-label {
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: var(--text-muted);
        font-weight: 600;
        margin-bottom: 0.25rem;
    }
    .hud-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.45rem;
        font-weight: 600;
        color: var(--text-main);
        line-height: 1.2;
    }
    .hud-sub {
        font-size: 0.78rem;
        color: var(--text-muted);
        margin-top: 0.2rem;
    }

    /* AI Co-Pilot Report Card */
    .copilot-card {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-left: 3px solid var(--accent);
        border-radius: 10px;
        padding: 1.3rem 1.5rem;
        margin-top: 0.75rem;
        color: var(--text-main);
    }

    /* Empty State Box */
    .empty-state {
        text-align: center;
        padding: 2.75rem 1.5rem;
        border: 1px dashed var(--border);
        border-radius: 10px;
        background: var(--bg-card);
        margin-top: 1rem;
    }
    .empty-state h3 {
        color: var(--text-main);
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 0.35rem;
    }
    .empty-state p {
        color: var(--text-muted);
        font-size: 0.9rem;
        margin: 0;
    }

    /* Crisp Image Frames */
    [data-testid="stImage"] img {
        border-radius: 8px;
        border: 1px solid var(--border);
    }

    /* Flat Enterprise Primary Button */
    .stButton > button[kind="primary"] {
        background: var(--accent);
        color: var(--btn-text);
        font-weight: 600;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 1.15rem;
        transition: background 0.15s ease;
    }
    .stButton > button[kind="primary"]:hover {
        background: var(--accent-hover);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

COLORMAPS = {
    "Turbo (High Contrast)": cv2.COLORMAP_TURBO,
    "Inferno (Thermal)": cv2.COLORMAP_INFERNO,
    "Ocean (Sonar Blue)": cv2.COLORMAP_OCEAN,
    "Jet (Classic)": cv2.COLORMAP_JET,
}


@st.cache_resource
def load_model(path: str = "sonar_ae.onnx"):
    return ort.InferenceSession(path, providers=["CPUExecutionProvider"])


def get_api_key() -> str:
    try:
        if "GEMINI_API_KEY" in st.secrets:
            return st.secrets["GEMINI_API_KEY"]
    except Exception:
        pass
    return ""


def analyze_sonar(
    pil_img: Image.Image,
    session: ort.InferenceSession,
    threshold_pct: float,
    cmap_code: int,
    overlay_alpha: float,
):
    t0 = time.perf_counter()
    orig_rgb = np.array(pil_img.convert("RGB").resize((256, 256)))
    gray = cv2.cvtColor(orig_rgb, cv2.COLOR_RGB2GRAY)
    gray_clean = cv2.medianBlur(gray, 3)

    raw_mask = (gray_clean > 8).astype(np.uint8)
    if raw_mask.sum() == 0:
        empty = Image.fromarray(orig_rgb)
        return empty, empty, empty, empty, {"detected": False, "latency_ms": 0.0}

    fan_mask = cv2.erode(raw_mask, np.ones((7, 7), np.uint8), iterations=1)
    if fan_mask.sum() == 0:
        fan_mask = raw_mask

    inp = (gray_clean.astype(np.float32) / 255.0)[None, None, :, :]
    input_name = session.get_inputs()[0].name
    recon = session.run(None, {input_name: inp})[0].squeeze()

    error_map = np.abs((gray_clean.astype(np.float32) / 255.0) - recon) * fan_mask
    error_smooth = cv2.GaussianBlur(error_map, (9, 9), 0)

    thresh = np.percentile(error_smooth[fan_mask > 0], threshold_pct)
    binary = (error_smooth > thresh).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    annotated = orig_rgb.copy()
    telemetry = {"detected": False}

    if contours:
        largest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)

        # Draw tactical corner brackets + crisp cyan bounding box
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 229, 255), 2)
        cv2.circle(annotated, (int(x + w / 2), int(y + h / 2)), 3, (0, 255, 170), -1)

        ys, xs = np.where(raw_mask > 0)
        apex_x, apex_y = int(np.mean(xs)), int(np.min(ys))
        max_radius = max(1, int(np.max(ys) - apex_y))

        cx, cy = x + w / 2, y + h / 2
        dx, dy = cx - apex_x, cy - apex_y
        range_pct = min(100.0, (math.hypot(dx, dy) / max_radius) * 100.0)
        bearing_deg = math.degrees(math.atan2(dx, max(1.0, dy)))
        side = (
            "Port (Left)"
            if bearing_deg < -3
            else ("Starboard (Right)" if bearing_deg > 3 else "Centerline")
        )

        telemetry = {
            "detected": True,
            "anomaly_score": float(error_smooth[y : y + h, x : x + w].max()),
            "range_percent": round(range_pct, 1),
            "bearing_deg": round(bearing_deg, 1),
            "side": side,
            "box_area_px": int(w * h),
        }

    telemetry["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)

    norm_heat = cv2.normalize(error_smooth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    heat_rgb = cv2.cvtColor(cv2.applyColorMap(norm_heat, cmap_code), cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(orig_rgb, 1.0 - overlay_alpha, heat_rgb, overlay_alpha, 0)
    recon_vis = cv2.cvtColor((np.clip(recon, 0, 1) * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)

    return (
        Image.fromarray(orig_rgb),
        Image.fromarray(overlay),
        Image.fromarray(annotated),
        Image.fromarray(recon_vis),
        telemetry,
    )


def render_hud_card(label: str, value: str, sub: str):
    st.markdown(
        f"""
        <div class="hud-card">
            <div class="hud-label">{label}</div>
            <div class="hud-value">{value}</div>
            <div class="hud-sub">{sub}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# --- Sidebar ---
with st.sidebar:
    st.markdown("### 🎛️ Sonar Control Panel")
    st.caption("Tune the unsupervised anomaly engine in real time.")

    threshold = st.slider(
        "Sensitivity Percentile",
        min_value=95.0,
        max_value=99.5,
        value=98.5,
        step=0.1,
        help="Higher percentile isolates only the sharpest bright acoustic echoes.",
    )
    cmap_label = st.selectbox("Heatmap Colormap", list(COLORMAPS.keys()), index=0)
    overlay_alpha = st.slider("Heatmap Opacity", 0.15, 0.75, 0.40, 0.05)

    st.divider()
    st.markdown("### 🔑 AI Co-Pilot Status")
    api_key = get_api_key()
    if api_key:
        st.success("Gemini API Ready (Secrets)", icon="✅")
    else:
        api_key = st.text_input("Gemini API Key", type="password", placeholder="AIza...")
        st.caption("Store GEMINI_API_KEY in `.streamlit/secrets.toml` or Cloud Secrets.")

    st.divider()
    st.markdown("### 📐 Pipeline Specs")
    st.caption(
        "• **Model:** Masked Conv-Autoencoder (ONNX)\n"
        "• **Training Mode:** Zero-Label Unsupervised\n"
        "• **Co-Pilot:** Gemini 3.5 Flash-Lite"
    )

# --- Hero Header ---
st.markdown(
    """
    <div class="hero-banner">
        <div class="status-pill"><span class="pulse-dot"></span>ONNX ACOUSTIC ENGINE • ONLINE</div>
        <h1 class="hero-title">📡 Forward-Looking Sonar Anomaly & AI Co-Pilot</h1>
        <p class="hero-sub">Zero-label unsupervised target localization, polar range/bearing telemetry, and automated maritime inspection logging.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

session = load_model()

uploaded = st.file_uploader(
    "Drop a Forward-Looking Sonar scan (.jpg / .png) to begin inspection",
    type=["jpg", "jpeg", "png"],
)

if uploaded is not None:
    pil_img = Image.open(uploaded)
    raw_img, heat_img, box_img, recon_img, telemetry = analyze_sonar(
        pil_img, session, threshold, COLORMAPS[cmap_label], overlay_alpha
    )

    if st.session_state.get("last_file") != uploaded.name:
        st.session_state["last_file"] = uploaded.name
        st.session_state["ai_report"] = None

    # --- HUD Telemetry Row ---
    if telemetry["detected"]:
        h1, h2, h3, h4 = st.columns(4)
        with h1:
            render_hud_card("Est. Target Range", f"{telemetry['range_percent']}%", "of max sonar swath")
        with h2:
            render_hud_card("Bearing Angle", f"{telemetry['bearing_deg']}°", telemetry["side"])
        with h3:
            render_hud_card("Peak Anomaly", f"{telemetry['anomaly_score']:.4f}", f"Area: {telemetry['box_area_px']} px²")
        with h4:
            render_hud_card("Inference Time", f"{telemetry['latency_ms']} ms", "CPU ONNX Runtime")
    else:
        st.warning("No high-confidence anomaly detected above the current sensitivity threshold. Try lowering the slider in the sidebar.")

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

    # --- Visual Inspection Tabs ---
    tab1, tab2 = st.tabs(["🎯 Tactical Overview", "🔬 Autoencoder Diagnostics"])

    with tab1:
        c1, c2, c3 = st.columns(3, gap="medium")
        with c1:
            st.markdown("**1. Raw FLS Scan**")
            st.image(raw_img, width="stretch")
        with c2:
            st.markdown("**2. Anomaly Heatmap**")
            st.image(heat_img, width="stretch")
        with c3:
            st.markdown("**3. Localized Target**")
            st.image(box_img, width="stretch")

    with tab2:
        d1, d2 = st.columns(2, gap="medium")
        with d1:
            st.markdown("**Autoencoder Background Reconstruction**")
            st.image(recon_img, width="stretch")
            st.caption("The model learns smooth seabed patterns; bright foreground targets fail to reconstruct and light up as anomalies.")
        with d2:
            st.markdown("**Blended Acoustic Error Map**")
            st.image(heat_img, width="stretch")
            st.caption(f"Thresholded at the {threshold}th percentile strictly inside the eroded sonar fan wedge.")

    # --- AI Inspection Co-Pilot Section ---
    if telemetry["detected"]:
        st.divider()
        col_title, col_btn = st.columns([2, 1], vertical_alignment="center")
        with col_title:
            st.markdown("### 🤖 AI Maritime Inspection Co-Pilot")
            st.caption("Generates an acoustic signature breakdown, object hypotheses, risk triage, and next ROV maneuver.")
        with col_btn:
            generate_clicked = st.button("✨ Generate Inspection Log", type="primary", width="stretch")

        if generate_clicked:
            if not api_key:
                st.error("Add your GEMINI_API_KEY in `.streamlit/secrets.toml` or paste it into the sidebar.")
            else:
                with st.spinner("Analyzing acoustic backscatter and shadow geometry with Gemini..."):
                    prompt = f"""
                    You are a Forward-Looking Sonar (FLS) maritime inspection co-pilot.
                    An unsupervised anomaly detector flagged the cyan-boxed region in this sonar scan.

                    Calculated Telemetry:
                    - Estimated Range: {telemetry['range_percent']}% of max sonar range
                    - Bearing: {telemetry['bearing_deg']} degrees ({telemetry['side']})
                    - Peak Reconstruction Anomaly Score: {telemetry['anomaly_score']:.4f}

                    Provide a crisp, structured inspection log with:
                    1. **Acoustic Signature** (shape, reflectivity, presence of acoustic shadow)
                    2. **Plausible Underwater Object Types** (be candid about uncertainty)
                    3. **Risk / Priority Level** (Low / Medium / High + 1-line rationale)
                    4. **Recommended Next ROV/AUV Maneuver**
                    """
                    try:
                        client = genai.Client(api_key=api_key)
                        response = client.models.generate_content(
                            model="gemini-3.5-flash-lite",
                            contents=[box_img, prompt],
                        )
                        st.session_state["ai_report"] = response.text
                    except Exception as e:
                        st.error(f"Gemini API error: {e}")

        if st.session_state.get("ai_report"):
            st.markdown(f'<div class="copilot-card">{st.session_state["ai_report"]}</div>', unsafe_allow_html=True)
            st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
            st.download_button(
                label="⬇️ Download Inspection Report (.md)",
                data=st.session_state["ai_report"],
                file_name=f"sonar_inspection_{uploaded.name}.md",
                mime="text/markdown",
            )
else:
    st.markdown(
        """
        <div class="empty-state">
            <h3>Ready for Acoustic Scan</h3>
            <p>Upload a <code>.jpg</code> or <code>.png</code> Forward-Looking Sonar frame above to run anomaly localization and AI inspection.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )