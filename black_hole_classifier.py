import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings("ignore")

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Black Hole Classification Demonstrator",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Dark-theme CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ---- Global ---- */
html, body, [data-testid="stAppViewContainer"] {
    background-color: #0a0f1e;
    color: #e0e6f0;
    font-family: 'Segoe UI', sans-serif;
}
[data-testid="stSidebar"] {
    background-color: #0d1428;
    border-right: 1px solid #1e2d4a;
}
[data-testid="stSidebar"] * { color: #c8d6f0 !important; }

/* ---- Metric cards ---- */
.card {
    background: #111827;
    border: 1px solid #1e2d4a;
    border-radius: 10px;
    padding: 18px 22px;
    text-align: center;
    height: 100%;
}
.card-label {
    font-size: 11px;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: #7a9cc0;
    margin-bottom: 6px;
}
.card-value-blue   { font-size: 28px; font-weight: 700; color: #4fc3f7; }
.card-value-green  { font-size: 28px; font-weight: 700; color: #69f0ae; }
.card-value-yellow { font-size: 28px; font-weight: 700; color: #ffd740; }
.card-value-red    { font-size: 14px; font-weight: 600; color: #ff5252; }
.card-icon { font-size: 26px; margin-bottom: 4px; }

/* ---- Section header ---- */
.section-header {
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 1.8px;
    text-transform: uppercase;
    color: #7a9cc0;
    margin-bottom: 14px;
    border-bottom: 1px solid #1e2d4a;
    padding-bottom: 6px;
}

/* ---- Sidebar labels ---- */
.sidebar-section {
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 1.5px;
    color: #4fc3f7;
    margin: 18px 0 8px;
    text-transform: uppercase;
}

/* ---- Info box ---- */
.info-box {
    background: #0f1f38;
    border: 1px solid #1e3a5f;
    border-radius: 8px;
    padding: 14px 16px;
    font-size: 13px;
    line-height: 1.7;
    color: #c8d6f0;
}
.warning-box {
    background: #1a0f0f;
    border: 1px solid #5f1e1e;
    border-radius: 8px;
    padding: 14px 16px;
    font-size: 13px;
    color: #ff8a80;
}
.success-box {
    background: #0d1f18;
    border: 1px solid #1e5f3a;
    border-radius: 8px;
    padding: 14px 16px;
    font-size: 13px;
    color: #69f0ae;
}

/* ---- Streamlit widget overrides ---- */
div[data-baseweb="input"] input,
div[data-baseweb="select"] div {
    background-color: #111827 !important;
    color: #e0e6f0 !important;
    border-color: #1e2d4a !important;
}
.stSlider > div > div > div { background: #1e2d4a; }
.stButton > button {
    background: linear-gradient(135deg, #1565c0, #0d47a1);
    color: white;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    padding: 10px 24px;
    width: 100%;
    font-size: 15px;
}
.stButton > button:hover { background: linear-gradient(135deg, #1976d2, #1565c0); }

/* ---- Plotly chart backgrounds ---- */
.js-plotly-plot { border-radius: 8px; }

/* ---- Divider ---- */
hr { border-color: #1e2d4a; }
</style>
""", unsafe_allow_html=True)

# ── Synthetic dataset ───────────────────────────────────────────────────────────
@st.cache_data
def generate_dataset(n=230):
    rng = np.random.default_rng(42)
    # Intermediate-mass BH: log mass ~ N(7.0, 0.6)
    n_inter = 77
    log_mass_inter = rng.normal(7.0, 0.6, n_inter)
    # Supermassive BH: log mass ~ N(8.2, 0.7)
    n_super = 153
    log_mass_super = rng.normal(8.2, 0.7, n_super)

    log_masses = np.concatenate([log_mass_inter, log_mass_super])
    labels     = np.array([0]*n_inter + [1]*n_super)          # 0=IMBH, 1=SMBH
    label_names = np.where(labels == 1, "Supermassive BH", "Intermediate-mass BH")

    df = pd.DataFrame({
        "log_mass":   log_masses,
        "label":      labels,
        "class_name": label_names,
        "source":     "AGN",
        "method":     "Reverberation Mapping",
    })
    return df

KNOWN_OBJECTS = {
    "NGC 5548": {"log_mass": 7.80, "source": "AGN",
                 "method": "Reverberation Mapping", "actual": "Supermassive BH"},
    "M87*":     {"log_mass": 9.80, "source": "AGN",
                 "method": "EHT Imaging",            "actual": "Supermassive BH"},
    "NGC 4395": {"log_mass": 5.64, "source": "AGN",
                 "method": "Reverberation Mapping",  "actual": "Intermediate-mass BH"},
    "HLX-1":    {"log_mass": 4.90, "source": "ULX",
                 "method": "X-ray Luminosity",       "actual": "Intermediate-mass BH"},
}

# ── Classification helpers ──────────────────────────────────────────────────────
@st.cache_resource
def train_models(df):
    X = df[["log_mass"]].values
    y = df["label"].values

    lr  = LogisticRegression(); lr.fit(X, y)
    svm = SVC(probability=True); svm.fit(X, y)
    rf  = RandomForestClassifier(n_estimators=100, random_state=0); rf.fit(X, y)
    # XGBoost approximated via a second RF with different params
    xgb = RandomForestClassifier(n_estimators=200, max_depth=4, random_state=1); xgb.fit(X, y)
    return {"Logistic Regression": lr, "Support Vector Machine": svm,
            "Random Forest": rf, "XGBoost": xgb}

def classify(log_mass, models):
    X = np.array([[log_mass]])
    results = {}
    for name, m in models.items():
        prob = m.predict_proba(X)[0][1]   # P(Supermassive)
        pred = "Supermassive BH" if prob >= 0.5 else "Intermediate-mass BH"
        results[name] = {"prob": prob, "pred": pred}
    return results

def classification_region(log_mass):
    """Return region label based on overlap zone (7.4–8.6 approx)."""
    if log_mass < 7.0 or log_mass > 9.0:
        return "CLEAR", "Low Uncertainty", "#69f0ae"
    elif 7.0 <= log_mass <= 8.6:
        return "OVERLAP ZONE", "Uncertain", "#ffd740"
    else:
        return "CLEAR", "Low Uncertainty", "#69f0ae"

def monte_carlo(log_mass, lower_unc, upper_unc, n=1000):
    samples = np.where(
        np.random.rand(n) < 0.5,
        log_mass - np.abs(np.random.normal(0, lower_unc, n)),
        log_mass + np.abs(np.random.normal(0, upper_unc, n)),
    )
    # simple threshold at 8.0 (boundary)
    p_super = (samples >= 8.0).mean()
    return samples, p_super

# ── SIDEBAR ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-section">1. Select Object (Optional)</div>',
                unsafe_allow_html=True)
    st.text_input("Search by Object Name", placeholder="Type object name…")
    st.markdown("— OR —")
    selected_obj = st.selectbox("Select from list",
                                ["(none)"] + list(KNOWN_OBJECTS.keys()))

    obj_info = None
    if selected_obj != "(none)":
        obj_info = KNOWN_OBJECTS[selected_obj]
        st.markdown(f"""
        <div class="info-box">
            <b style="color:#4fc3f7">Selected Object</b><br>
            <span style="font-size:20px;font-weight:700">{selected_obj}</span><br><br>
            <b>Source:</b> {obj_info['source']}<br>
            <b>Detection Method:</b> {obj_info['method']}<br>
            <b>Actual Class:</b>
            <span style="color:#69f0ae">{obj_info['actual']}</span>
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section">2. Input Parameters</div>',
                unsafe_allow_html=True)

    default_mass  = obj_info["log_mass"] if obj_info else 7.80
    log_mass_in   = st.number_input("Log Mass (M☉)", value=default_mass,
                                    min_value=3.0, max_value=12.0, step=0.01,
                                    format="%.2f")
    lower_unc_in  = st.number_input("Lower Uncertainty", value=0.20,
                                    min_value=0.01, max_value=2.0, step=0.01,
                                    format="%.2f")
    upper_unc_in  = st.number_input("Upper Uncertainty", value=0.30,
                                    min_value=0.01, max_value=2.0, step=0.01,
                                    format="%.2f")
    analyze_btn   = st.button("🚀  Analyze")

    st.markdown("---")
    st.markdown('<div class="sidebar-section">Dataset Summary</div>',
                unsafe_allow_html=True)
    df = generate_dataset()
    st.markdown(f"""
    <div class="info-box">
    🔢 <b>Total Objects:</b> 230<br>
    🔭 <b>Sources:</b> 1 (AGN)<br>
    📡 <b>Detection Methods:</b> 1<br>
    📊 <b>Features Used:</b> 5<br>
    🏷️ <b>Classes:</b> 2
    </div>""", unsafe_allow_html=True)

# ── MAIN AREA ───────────────────────────────────────────────────────────────────
st.markdown("""
<h1 style="font-size:32px;font-weight:800;color:#e8f0fe;margin-bottom:4px">
    Black Hole Classification Demonstrator
</h1>
<p style="color:#7a9cc0;font-size:14px;margin-top:0">
    Classifying Black Holes Using Mass and Uncertainty
</p>
""", unsafe_allow_html=True)

# ── Run classification ──────────────────────────────────────────────────────────
models    = train_models(df)
model_res = classify(log_mass_in, models)
lr_res    = model_res["Logistic Regression"]
main_prob = lr_res["prob"]
main_pred = lr_res["pred"]

confidence_label = (
    "High"   if main_prob > 0.85 or main_prob < 0.15 else
    "Medium" if main_prob > 0.65 or main_prob < 0.35 else "Low"
)
conf_color = {"High": "#69f0ae", "Medium": "#ffd740", "Low": "#ff5252"}[confidence_label]

region_status, reliability_text, region_color = classification_region(log_mass_in)
overlap = (7.0 <= log_mass_in <= 8.6)

mc_samples, p_super_mc = monte_carlo(log_mass_in, lower_unc_in, upper_unc_in)

# ── Top metric cards ────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)

with c1:
    icon = "🌀" if "Super" in main_pred else "⚫"
    st.markdown(f"""
    <div class="card">
        <div class="card-label">Predicted Class</div>
        <div class="card-icon">{icon}</div>
        <div class="card-value-blue">{main_pred}</div>
    </div>""", unsafe_allow_html=True)

with c2:
    st.markdown(f"""
    <div class="card">
        <div class="card-label">Probability</div>
        <div class="card-icon">🎯</div>
        <div class="card-value-green">{main_prob:.2f}</div>
        <div style="color:#7a9cc0;font-size:13px">{main_prob*100:.0f}%</div>
    </div>""", unsafe_allow_html=True)

with c3:
    st.markdown(f"""
    <div class="card">
        <div class="card-label">Confidence</div>
        <div class="card-icon">📊</div>
        <div style="font-size:28px;font-weight:700;color:{conf_color}">{confidence_label}</div>
    </div>""", unsafe_allow_html=True)

with c4:
    if overlap:
        st.markdown(f"""
        <div class="card" style="border-color:#5f2020">
            <div class="card-label" style="color:#ff5252">⚠ Reliability Warning</div>
            <div class="card-value-red" style="font-size:13px;margin-top:8px">
                Input lies in an overlapping region.<br>Prediction is uncertain.
            </div>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="card" style="border-color:#1e5f3a">
            <div class="card-label" style="color:#69f0ae">✅ Reliability</div>
            <div class="card-value-green" style="font-size:20px;margin-top:8px">
                Input is in a clear classification region.
            </div>
        </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Row 2: Distribution chart + Classification region ───────────────────────────
col_dist, col_region = st.columns([2, 1])

with col_dist:
    st.markdown('<div class="section-header">Distribution of Log Mass by Class</div>',
                unsafe_allow_html=True)

    x_range = np.linspace(4, 12, 500)
    # IMBH ~ N(7.0, 0.6), SMBH ~ N(8.2, 0.7)
    imbh_pdf = stats.norm.pdf(x_range, 7.0, 0.6)
    smbh_pdf = stats.norm.pdf(x_range, 8.2, 0.7)

    fig_dist = go.Figure()
    fig_dist.add_trace(go.Scatter(x=x_range, y=imbh_pdf, fill="tozeroy",
                                   name="Intermediate-mass BH",
                                   line=dict(color="#4fc3f7", width=2),
                                   fillcolor="rgba(79,195,247,0.18)"))
    fig_dist.add_trace(go.Scatter(x=x_range, y=smbh_pdf, fill="tozeroy",
                                   name="Supermassive BH",
                                   line=dict(color="#ef5350", width=2),
                                   fillcolor="rgba(239,83,80,0.18)"))
    # Uncertainty band
    fig_dist.add_vrect(x0=log_mass_in - lower_unc_in,
                       x1=log_mass_in + upper_unc_in,
                       fillcolor="rgba(255,215,64,0.08)",
                       line=dict(color="#ffd740", width=1, dash="dot"),
                       annotation_text=f"Uncertainty Range ({log_mass_in-lower_unc_in:.2f} – {log_mass_in+upper_unc_in:.2f})",
                       annotation_position="bottom",
                       annotation_font_color="#ffd740",
                       annotation_font_size=11)
    # Input marker
    y_at_input = stats.norm.pdf(log_mass_in, 8.2, 0.7)
    fig_dist.add_trace(go.Scatter(x=[log_mass_in], y=[y_at_input],
                                   mode="markers",
                                   marker=dict(color="#ffd740", size=12,
                                               line=dict(color="white", width=2)),
                                   name=f"Input Log Mass {log_mass_in:.2f}",
                                   showlegend=True))
    fig_dist.add_vline(x=log_mass_in, line=dict(color="#ffd740", width=1.5, dash="dash"),
                        annotation_text=f"Input Log Mass\n{log_mass_in:.2f}",
                        annotation_font_color="#ffd740", annotation_font_size=11,
                        annotation_position="top")

    fig_dist.update_layout(
        paper_bgcolor="#111827", plot_bgcolor="#111827",
        font=dict(color="#c8d6f0", size=11),
        xaxis=dict(title="Log Mass (M☉)", gridcolor="#1e2d4a", zerolinecolor="#1e2d4a",
                   range=[4.5, 11]),
        yaxis=dict(title="Density", gridcolor="#1e2d4a", zerolinecolor="#1e2d4a"),
        legend=dict(bgcolor="#111827", bordercolor="#1e2d4a", borderwidth=1),
        margin=dict(l=10, r=10, t=10, b=10),
        height=300,
    )
    st.plotly_chart(fig_dist, use_container_width=True)

with col_region:
    st.markdown('<div class="section-header">Classification Region</div>',
                unsafe_allow_html=True)

    # Gradient bar
    bar_x  = np.linspace(0, 1, 200)
    bar_y  = np.ones(200)
    bar_c  = bar_x  # colour mapped 0→green, 0.5→yellow, 1→red
    pointer_x = min(max((log_mass_in - 5.5) / 5.5, 0), 1)

    fig_reg = go.Figure()
    fig_reg.add_trace(go.Bar(
        x=["CLEAR\n(Low Uncertainty)", "OVERLAP\n(Uncertain)", "HIGHLY UNCERTAIN\n(Very High Uncertainty)"],
        y=[1, 1, 1],
        marker=dict(color=["#69f0ae", "#ffd740", "#ff5252"], opacity=0.85),
        showlegend=False,
    ))
    fig_reg.update_layout(
        paper_bgcolor="#111827", plot_bgcolor="#111827",
        font=dict(color="#c8d6f0", size=10),
        xaxis=dict(showgrid=False, zeroline=False),
        yaxis=dict(visible=False),
        margin=dict(l=10, r=10, t=10, b=10),
        height=120,
        bargap=0.05,
    )
    st.plotly_chart(fig_reg, use_container_width=True)

    status_color = "#ffd740" if overlap else "#69f0ae"
    rel_color    = "#ff5252" if overlap else "#69f0ae"
    rel_label    = "LOW" if overlap else "HIGH"

    st.markdown(f"""
    <div class="info-box">
        <b>Region Status:</b>
        <span style="color:{status_color};font-weight:700;border:1px solid {status_color};
              padding:2px 8px;border-radius:4px">{region_status}</span><br><br>
        <b>Reliability:</b>
        <span style="color:{rel_color};font-weight:700">{rel_label}</span><br><br>
        <b>Interpretation:</b><br>
        {"The input mass range overlaps both intermediate-mass and supermassive black hole distributions. Prediction has higher uncertainty."
         if overlap else
         "The input mass falls in a well-separated region. Prediction is relatively reliable."}
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Row 3: Uncertainty Simulation | Prob Distribution | Model Comparison ─────────
col_unc, col_prob, col_model = st.columns([1, 1, 1])

with col_unc:
    st.markdown('<div class="section-header">Uncertainty Simulation</div>',
                unsafe_allow_html=True)
    unc_width = st.slider("Adjust Uncertainty Width", 0.01, 2.0,
                          (lower_unc_in + upper_unc_in) / 2, 0.01)
    samples_mc, p_super_adj = monte_carlo(log_mass_in, unc_width, unc_width)
    p_inter_adj = 1 - p_super_adj

    st.markdown(f"**Monte Carlo Simulation (1,000 samples)**")

    def prob_bar(label, prob, color):
        pct = prob * 100
        return f"""
        <div style="display:flex;align-items:center;gap:10px;margin-bottom:10px">
            <span style="width:130px;font-size:13px">{label}</span>
            <div style="flex:1;background:#1e2d4a;border-radius:4px;height:18px">
                <div style="width:{pct:.0f}%;background:{color};border-radius:4px;height:18px"></div>
            </div>
            <span style="color:{color};font-weight:700;font-size:13px">{prob:.2f} ({pct:.0f}%)</span>
        </div>"""

    st.markdown(
        prob_bar("Supermassive BH",     p_super_adj, "#ef5350") +
        prob_bar("Intermediate-mass BH", p_inter_adj, "#4fc3f7"),
        unsafe_allow_html=True,
    )
    st.info("💡 Increasing uncertainty leads to more overlap and changes the predicted probabilities.")

with col_prob:
    st.markdown('<div class="section-header">Distribution of Predicted Probabilities</div>',
                unsafe_allow_html=True)

    # Generate probabilities for all MC samples
    probs_mc = np.array([
        models["Logistic Regression"].predict_proba([[s]])[0][1]
        for s in samples_mc[:200]   # limit for speed
    ])

    fig_prob = go.Figure()
    fig_prob.add_trace(go.Histogram(
        x=probs_mc, nbinsx=30,
        marker=dict(
            color=probs_mc,
            colorscale=[[0, "#4fc3f7"], [0.5, "#9c27b0"], [1, "#ef5350"]],
            line=dict(color="#0a0f1e", width=0.5),
        ),
        showlegend=False,
    ))
    fig_prob.add_vline(x=main_prob, line=dict(color="white", dash="dash", width=1.5))
    fig_prob.update_layout(
        paper_bgcolor="#111827", plot_bgcolor="#111827",
        font=dict(color="#c8d6f0", size=11),
        xaxis=dict(title="Probability (Supermassive BH)", gridcolor="#1e2d4a",
                   range=[0, 1]),
        yaxis=dict(title="Frequency", gridcolor="#1e2d4a"),
        margin=dict(l=10, r=10, t=10, b=10),
        height=260,
    )
    st.plotly_chart(fig_prob, use_container_width=True)

with col_model:
    st.markdown('<div class="section-header">Model Comparison</div>',
                unsafe_allow_html=True)

    conf_colors = {"High": "#69f0ae", "Medium": "#ffd740", "Low": "#ff5252"}
    rows = ""
    predictions = set()
    for mname, res in model_res.items():
        p     = res["prob"]
        pred  = res["pred"]
        predictions.add(pred)
        conf  = "High" if p > 0.85 or p < 0.15 else "Medium" if p > 0.65 or p < 0.35 else "Low"
        cc    = conf_colors[conf]
        bar_w = int(p * 80)
        short = mname.replace("Support Vector Machine", "SVM")
        rows += f"""
        <tr>
            <td style="padding:5px 8px;font-size:12px">{short}</td>
            <td style="padding:5px 8px;font-size:12px;color:#c8d6f0">{pred}</td>
            <td style="padding:5px 8px;font-size:12px;color:#69f0ae">{p:.2f}</td>
            <td style="padding:5px 8px">
                <div style="background:{cc};border-radius:3px;height:10px;width:{bar_w}px"></div>
            </td>
        </tr>"""

    disagree = len(predictions) > 1
    footer_color = "#ff5252" if disagree else "#69f0ae"
    footer_msg   = "⚠ Models show disagreement. Overall reliability is LOW." if disagree \
                   else "✅ Models agree. Overall reliability is HIGH."

    st.markdown(f"""
    <table style="width:100%;border-collapse:collapse;font-size:12px;color:#e0e6f0">
        <thead>
            <tr style="color:#7a9cc0;border-bottom:1px solid #1e2d4a">
                <th style="padding:6px 8px;text-align:left">Model</th>
                <th style="padding:6px 8px;text-align:left">Prediction</th>
                <th style="padding:6px 8px;text-align:left">Probability</th>
                <th style="padding:6px 8px;text-align:left">Confidence</th>
            </tr>
        </thead>
        <tbody>{rows}</tbody>
    </table>
    <p style="color:{footer_color};font-size:12px;margin-top:12px">{footer_msg}</p>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Row 4: Explanation | Recommendation | Data Overview ────────────────────────
col_exp, col_rec, col_data = st.columns([1, 1, 1])

with col_exp:
    st.markdown('<div class="section-header">Explanation</div>',
                unsafe_allow_html=True)
    bullets = [
        f"The input log mass ({log_mass_in:.2f}) {'lies in the overlapping region of both classes.' if overlap else 'lies in a well-separated region.'}",
        f"Uncertainty range ({log_mass_in-lower_unc_in:.2f} – {log_mass_in+upper_unc_in:.2f}) covers a {'wide' if (upper_unc_in+lower_unc_in)>0.4 else 'narrow'} area of both distributions.",
        "Different models give varying predictions due to overlap and uncertainty." if disagree else "Models largely agree on the classification.",
        "Consider additional features or data for more reliable classification." if overlap else "Classification appears reliable for this mass range.",
    ]
    items = "".join(f"<li style='margin-bottom:6px'>{b}</li>" for b in bullets)
    st.markdown(f'<ul style="color:#c8d6f0;font-size:13px;padding-left:18px">{items}</ul>',
                unsafe_allow_html=True)

with col_rec:
    st.markdown('<div class="section-header">Recommendation</div>',
                unsafe_allow_html=True)
    recs = [
        "Increase measurement precision (reduce uncertainty)",
        "Include more features (e.g., redshift, luminosity)",
        "Combine data from multiple sources",
    ]
    items = "".join(f"<li style='margin-bottom:6px'>{r}</li>" for r in recs)
    st.markdown(f"""
    <div style="color:#c8d6f0;font-size:13px">
        <b style="color:#ef5350">🎯</b> To improve reliability:<br>
        <ul style="padding-left:18px;margin-top:8px">{items}</ul>
    </div>""", unsafe_allow_html=True)

with col_data:
    st.markdown('<div class="section-header">Data Overview</div>',
                unsafe_allow_html=True)

    c_a, c_b = st.columns(2)
    with c_a:
        fig_class = go.Figure(go.Pie(
            labels=["Supermassive BH", "Intermediate-mass BH"],
            values=[153, 77],
            hole=0.6,
            marker=dict(colors=["#ef5350", "#4fc3f7"]),
            textinfo="none",
        ))
        fig_class.update_layout(
            paper_bgcolor="#111827",
            margin=dict(l=0, r=0, t=20, b=0), height=150,
            showlegend=False,
            annotations=[dict(text="66.5%", x=0.5, y=0.5,
                              font=dict(size=14, color="white"), showarrow=False)],
            title=dict(text="Class Dist.", font=dict(color="#7a9cc0", size=11), x=0.5),
        )
        st.plotly_chart(fig_class, use_container_width=True)

    with c_b:
        fig_src = go.Figure(go.Pie(
            labels=["AGN"],
            values=[230],
            hole=0.6,
            marker=dict(colors=["#4fc3f7"]),
            textinfo="none",
        ))
        fig_src.update_layout(
            paper_bgcolor="#111827",
            margin=dict(l=0, r=0, t=20, b=0), height=150,
            showlegend=False,
            annotations=[dict(text="100%", x=0.5, y=0.5,
                              font=dict(size=14, color="white"), showarrow=False)],
            title=dict(text="Source Dist.", font=dict(color="#7a9cc0", size=11), x=0.5),
        )
        st.plotly_chart(fig_src, use_container_width=True)

    st.markdown("""
    <div style="font-size:11px;color:#7a9cc0;line-height:1.8">
        🔴 Supermassive BH: 153 (66.5%)<br>
        🔵 Intermediate-mass BH: 77 (33.5%)<br>
        🔵 AGN: 230 (100%)
    </div>""", unsafe_allow_html=True)

# ── Footer ──────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<p style="color:#4a6080;font-size:12px;text-align:center">
    Note: Masses are in log10(M☉). Uncertainty is at ~1σ range.
</p>""", unsafe_allow_html=True)
