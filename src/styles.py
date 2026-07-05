"""RespiGuard-AI Design System.

Clinical Intelligence theme: dark, data-forward, trustworthy, medical precision.
Signature element: glowing pulsing red badge on the CRITICAL risk level.

This module holds the full CSS string and all reusable Streamlit UI render
helpers so that every page shares one consistent visual identity.
"""

# ---------------------------------------------------------------------------
# Color palette (single source of truth for the whole app)
# ---------------------------------------------------------------------------
COLORS = {
    "bg_primary": "#0A1628",
    "bg_card": "#1E2D40",
    "bg_sidebar": "#112035",
    "accent_primary": "#00D4AA",
    "accent_secondary": "#0EA5E9",
    "text_primary": "#F1F5F9",
    "text_secondary": "#94A3B8",
    "border": "#2D4A6B",
    "risk_low": "#22C55E",
    "risk_lowmedium": "#84CC16",
    "risk_medium": "#EAB308",
    "risk_high": "#F97316",
    "risk_critical": "#EF4444",
}

# Map a normalized risk level string to the palette color.
RISK_COLOR_MAP = {
    "LOW": COLORS["risk_low"],
    "LOW-MEDIUM": COLORS["risk_lowmedium"],
    "MEDIUM": COLORS["risk_medium"],
    "HIGH": COLORS["risk_high"],
    "CRITICAL": COLORS["risk_critical"],
}

# Map a normalized risk level string to its CSS badge class.
RISK_BADGE_CLASS = {
    "LOW": "risk-low",
    "LOW-MEDIUM": "risk-lowmedium",
    "MEDIUM": "risk-medium",
    "HIGH": "risk-high",
    "CRITICAL": "risk-critical",
}

# Map a normalized risk level string to its recommendation-box CSS class.
RISK_REC_CLASS = {
    "LOW": "rec-low",
    "LOW-MEDIUM": "rec-low",
    "MEDIUM": "rec-medium",
    "HIGH": "rec-high",
    "CRITICAL": "rec-critical",
}


def load_css() -> str:
    """Return the complete CSS string for the RespiGuard-AI app.

    Includes Google Font imports, base theme, sidebar styling, cards,
    risk badges (with the pulsing red CRITICAL animation), recommendation
    boxes, forecast cards, and Streamlit widget overrides.
    """
    return """
    <style>
    /* ----- Google Fonts ----- */
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500;600&display=swap');

    /* ----- Base body / app background ----- */
    .stApp, body {
        background-color: #0A1628;
        color: #F1F5F9;
        font-family: 'Inter', sans-serif;
    }
    .block-container {
        padding-top: 2rem;
        max-width: 1200px;
    }

    /* ----- Sidebar ----- */
    section[data-testid="stSidebar"] {
        background-color: #112035;
        border-right: 1px solid #2D4A6B;
    }
    section[data-testid="stSidebar"] .stRadio label {
        color: #94A3B8;
        font-family: 'Inter', sans-serif;
        transition: color 0.2s ease;
    }
    section[data-testid="stSidebar"] .stRadio label:hover {
        color: #00D4AA;
    }

    /* ----- Headers ----- */
    .main-header {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 2.5rem;
        font-weight: 700;
        color: #F1F5F9;
        line-height: 1.1;
    }
    .main-subtitle {
        font-family: 'Inter', sans-serif;
        font-size: 1rem;
        color: #94A3B8;
        margin-top: 4px;
    }
    .teal-accent {
        color: #00D4AA;
    }

    /* ----- Metric card ----- */
    .metric-card {
        background-color: #1E2D40;
        border: 1px solid #2D4A6B;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        transition: border-color 0.3s ease;
    }
    .metric-card:hover {
        border-color: #00D4AA;
        transition: 0.3s ease;
    }
    .card-label {
        font-family: 'Inter', sans-serif;
        font-size: 0.8rem;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .card-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 2rem;
        font-weight: 600;
        color: #00D4AA;
    }
    .card-unit {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1rem;
        color: #94A3B8;
        margin-left: 4px;
    }

    /* ----- Risk badges ----- */
    .risk-badge {
        display: inline-block;
        padding: 6px 16px;
        border-radius: 20px;
        font-family: 'Space Grotesk', sans-serif;
        font-size: 0.9rem;
        font-weight: 600;
        text-transform: uppercase;
        border: 1px solid;
    }
    .risk-low {
        background: rgba(34,197,94,0.15);
        color: #22C55E;
        border-color: #22C55E;
    }
    .risk-lowmedium {
        background: rgba(132,204,22,0.15);
        color: #84CC16;
        border-color: #84CC16;
    }
    .risk-medium {
        background: rgba(234,179,8,0.15);
        color: #EAB308;
        border-color: #EAB308;
    }
    .risk-high {
        background: rgba(249,115,22,0.15);
        color: #F97316;
        border-color: #F97316;
    }
    .risk-critical {
        background: rgba(239,68,68,0.15);
        color: #EF4444;
        border-color: #EF4444;
        animation: pulse-red 1.5s infinite;
    }
    @keyframes pulse-red {
        0%   { box-shadow: 0 0 0 0 rgba(239,68,68,0.6); }
        70%  { box-shadow: 0 0 0 12px rgba(239,68,68,0); }
        100% { box-shadow: 0 0 0 0 rgba(239,68,68,0); }
    }

    /* ----- Recommendation box ----- */
    .recommendation-box {
        border-radius: 10px;
        padding: 1.2rem 1.5rem;
        margin-top: 1rem;
        border-left: 4px solid;
        background-color: #1E2D40;
        font-family: 'Inter', sans-serif;
        color: #F1F5F9;
    }
    .rec-critical { border-left-color: #EF4444; background: rgba(239,68,68,0.08); }
    .rec-high     { border-left-color: #F97316; background: rgba(249,115,22,0.08); }
    .rec-medium   { border-left-color: #EAB308; background: rgba(234,179,8,0.08); }
    .rec-low      { border-left-color: #22C55E; background: rgba(34,197,94,0.08); }

    /* ----- Dividers ----- */
    .section-divider {
        border: none;
        border-top: 1px solid #2D4A6B;
        margin: 2rem 0;
    }

    /* ----- AQI display ----- */
    .aqi-number {
        font-family: 'JetBrains Mono', monospace;
        font-size: 3.5rem;
        font-weight: 600;
        line-height: 1;
    }
    .aqi-label {
        font-family: 'Inter', sans-serif;
        font-size: 0.75rem;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* ----- Forecast cards ----- */
    .forecast-container {
        display: flex;
        gap: 1rem;
        margin-top: 1rem;
    }
    .forecast-bar-item {
        flex: 1;
        background: #1E2D40;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        border: 1px solid #2D4A6B;
    }
    .forecast-time {
        font-family: 'Inter', sans-serif;
        font-size: 0.8rem;
        color: #94A3B8;
        margin-bottom: 8px;
    }
    .forecast-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.5rem;
        font-weight: 600;
    }

    /* ----- Sidebar branding ----- */
    .sidebar-logo {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.4rem;
        font-weight: 700;
        color: #00D4AA;
        padding: 1rem 0 0.5rem;
        text-align: center;
    }
    .sidebar-tagline {
        font-family: 'Inter', sans-serif;
        font-size: 0.75rem;
        color: #94A3B8;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .sidebar-footer {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.75rem;
        color: #94A3B8;
        text-align: center;
        margin-top: 1.5rem;
    }

    /* ----- Version badge ----- */
    .version-badge {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.75rem;
        background: rgba(0,212,170,0.15);
        color: #00D4AA;
        border: 1px solid #00D4AA;
        border-radius: 4px;
        padding: 2px 8px;
    }

    /* ----- Data mode banner ----- */
    .data-mode-banner {
        background: rgba(14,165,233,0.1);
        border: 1px solid #0EA5E9;
        border-radius: 8px;
        padding: 1rem;
        color: #7DD3FC;
        font-family: 'Inter', sans-serif;
        font-size: 0.875rem;
    }
    .data-mode-banner-real {
        background: rgba(34,197,94,0.1);
        border: 1px solid #22C55E;
        border-radius: 8px;
        padding: 1rem;
        color: #86EFAC;
        font-family: 'Inter', sans-serif;
        font-size: 0.875rem;
    }

    /* ----- Data table ----- */
    .risk-table {
        width: 100%;
        border-collapse: collapse;
        font-family: 'Inter', sans-serif;
        margin-top: 1rem;
    }
    .risk-table th {
        text-align: left;
        padding: 0.75rem 1rem;
        color: #94A3B8;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        border-bottom: 1px solid #2D4A6B;
    }
    .risk-table td {
        padding: 0.75rem 1rem;
        color: #F1F5F9;
        border-bottom: 1px solid #2D4A6B;
        font-size: 0.9rem;
    }
    .risk-table td.mono {
        font-family: 'JetBrains Mono', monospace;
    }

    /* ----- Streamlit buttons ----- */
    .stButton > button {
        background-color: #00D4AA;
        color: #0A1628;
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 600;
        border-radius: 8px;
        border: none;
        width: 100%;
        transition: all 0.2s ease;
    }
    .stButton > button:hover {
        background-color: #00B896;
        transform: translateY(-1px);
        color: #0A1628;
    }

    /* ----- Streamlit selectbox ----- */
    div[data-baseweb="select"] > div {
        background-color: #1E2D40;
        border: 1px solid #2D4A6B;
        color: #F1F5F9;
        border-radius: 8px;
    }

    /* ----- Streamlit metric ----- */
    div[data-testid="stMetricValue"] {
        font-family: 'JetBrains Mono', monospace;
        color: #00D4AA;
    }
    div[data-testid="stMetricLabel"] {
        font-family: 'Inter', sans-serif;
        color: #94A3B8;
    }

    /* ----- Hide default Streamlit chrome ----- */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header { visibility: hidden; }
    </style>
    """


def apply_styles():
    """Inject the RespiGuard-AI CSS into the current Streamlit app."""
    import streamlit as st

    st.markdown(load_css(), unsafe_allow_html=True)


def apply_graph_theme():
    """Apply a dark matplotlib theme that matches the app colors."""
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.facecolor": "#0A1628",
        "axes.facecolor": "#1E2D40",
        "axes.edgecolor": "#2D4A6B",
        "axes.labelcolor": "#94A3B8",
        "text.color": "#F1F5F9",
        "xtick.color": "#94A3B8",
        "ytick.color": "#94A3B8",
        "grid.color": "#2D4A6B",
        "grid.alpha": 0.5,
        "figure.dpi": 150,
    })


def _normalize_risk(risk_level: str) -> str:
    """Normalize a risk-level string to the canonical uppercase form."""
    return str(risk_level).strip().upper()


def render_risk_badge(risk_level: str):
    """Render a colored (and animated for CRITICAL) risk badge in Streamlit."""
    import streamlit as st

    level = _normalize_risk(risk_level)
    css_class = RISK_BADGE_CLASS.get(level, "risk-low")
    st.markdown(
        f'<span class="risk-badge {css_class}">{level}</span>',
        unsafe_allow_html=True,
    )


def render_recommendation_box(text: str, risk_level: str):
    """Render a left-border colored recommendation box matching the risk."""
    import streamlit as st

    level = _normalize_risk(risk_level)
    css_class = RISK_REC_CLASS.get(level, "rec-low")
    st.markdown(
        f'<div class="recommendation-box {css_class}">{text}</div>',
        unsafe_allow_html=True,
    )


def render_page_header(title: str, subtitle: str = None, version: str = None):
    """Render the branded page header with the teal 'Respi' accent."""
    import streamlit as st

    version_html = ""
    if version:
        version_html = f' <span class="version-badge">{version}</span>'

    subtitle_html = ""
    if subtitle:
        subtitle_html = f'<div class="main-subtitle">{subtitle}</div>'

    st.markdown(
        f"""
        <div class="main-header">
            <span class="teal-accent">Respi</span>Guard-AI{version_html}
        </div>
        <div class="main-subtitle">{title}</div>
        {subtitle_html}
        <hr class="section-divider">
        """,
        unsafe_allow_html=True,
    )


def render_aqi_display(aqi_value: int, city: str):
    """Render a large AQI number card, color coded by severity."""
    import streamlit as st

    aqi_value = int(round(aqi_value))
    # Color threshold: green < 100, yellow < 200, orange < 300, red >= 300
    if aqi_value < 100:
        color = COLORS["risk_low"]
    elif aqi_value < 200:
        color = COLORS["risk_medium"]
    elif aqi_value < 300:
        color = COLORS["risk_high"]
    else:
        color = COLORS["risk_critical"]

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="aqi-label">Current AQI &middot; {city}</div>
            <div class="aqi-number" style="color:{color};">{aqi_value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar(pages: list) -> str:
    """Render the branded sidebar and return the selected page name."""
    import streamlit as st

    st.sidebar.markdown(
        '<div class="sidebar-logo">🫁 RespiGuard-AI</div>',
        unsafe_allow_html=True,
    )
    st.sidebar.markdown(
        '<div class="sidebar-tagline">Hospital Early Warning System</div>',
        unsafe_allow_html=True,
    )
    st.sidebar.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    selected = st.sidebar.radio("Navigation", pages, label_visibility="collapsed")

    st.sidebar.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    return selected


def render_metric_card(label: str, value, unit: str = ""):
    """Render a styled metric card with an uppercase label and mono value."""
    import streamlit as st

    unit_html = f'<span class="card-unit">{unit}</span>' if unit else ""
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="card-label">{label}</div>
            <div class="card-value">{value}{unit_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_data_mode_banner(using_sample: bool):
    """Show a blue info banner for sample data or a green banner for real data."""
    import streamlit as st

    if using_sample:
        st.markdown(
            """
            <div class="data-mode-banner">
                🔵 <b>Sample Data Mode</b> &mdash; Running on auto-generated sample data.
                Upload your real CSV files to the <code>data/</code> folder
                (<code>aqi_data.csv</code>, <code>hospital_data.csv</code>,
                <code>weather_data.csv</code>) to switch to real data automatically.
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div class="data-mode-banner-real">
                🟢 <b>Real Data Active</b> &mdash; Predictions are running on the
                real datasets found in the <code>data/</code> folder.
            </div>
            """,
            unsafe_allow_html=True,
        )
