import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import folium
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from streamlit_folium import st_folium
import pandas as pd
import numpy as np
from analysis import (
    load_data, get_kpis, worst_areas, worst_roads, congestion_by_area,
    weather_impact, day_patterns, monthly_trend, roadwork_impact,
    correlation_data, map_data, generate_insights, train_model, predict_congestion,
    AREA_COORDS
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Bangalore Traffic Analyzer",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .main { background-color: #0e1117; }

    .hero-title {
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(135deg, #ff4b4b, #ff8c00, #ffd700);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .hero-sub {
        color: #888;
        font-size: 1rem;
        margin-bottom: 2rem;
    }

    .kpi-card {
        background: linear-gradient(135deg, #1a1d27, #22263a);
        border: 1px solid #2e3250;
        border-radius: 16px;
        padding: 1.4rem 1.6rem;
        text-align: center;
        transition: transform 0.2s;
    }
    .kpi-card:hover { transform: translateY(-3px); }
    .kpi-label { color: #888; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 1px; }
    .kpi-value { color: #ffffff; font-size: 1.9rem; font-weight: 700; margin: 0.3rem 0; }
    .kpi-sub   { color: #ff6b35; font-size: 0.8rem; font-weight: 600; }

    .insight-card {
        background: linear-gradient(135deg, #1a1d27, #22263a);
        border-left: 4px solid #ff4b4b;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.8rem;
        color: #ddd;
        font-size: 0.95rem;
    }

    .section-title {
        font-size: 1.3rem;
        font-weight: 600;
        color: #ffffff;
        margin: 2rem 0 1rem 0;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    .predict-card {
        background: linear-gradient(135deg, #1a1d27, #22263a);
        border: 1px solid #2e3250;
        border-radius: 16px;
        padding: 1.5rem;
    }

    .pred-result-high   { color: #ff4b4b; font-size: 3rem; font-weight: 700; }
    .pred-result-medium { color: #ff8c00; font-size: 3rem; font-weight: 700; }
    .pred-result-low    { color: #00d084; font-size: 3rem; font-weight: 700; }

    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #12151f, #1a1d2e);
        border-right: 1px solid #2e3250;
    }
    div[data-testid="stSidebar"] .stSelectbox label,
    div[data-testid="stSidebar"] .stMultiSelect label { color: #aaa !important; }

    .stTabs [data-baseweb="tab"] {
        background: transparent;
        color: #888;
        border-radius: 8px 8px 0 0;
        padding: 0.6rem 1.2rem;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #ff4b4b22, #ff8c0011) !important;
        color: #ff6b35 !important;
        border-bottom: 2px solid #ff4b4b !important;
    }

    .stMetric { background: #1a1d27; border-radius: 10px; padding: 1rem; }
</style>
""", unsafe_allow_html=True)

# ── Load data & train model ───────────────────────────────────────────────────
@st.cache_data
def get_data():
    return load_data()

@st.cache_resource
def get_model(df):
    return train_model(df)

df_full = get_data()
model, le_area, le_weather, le_road, model_score, features = get_model(df_full)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🚦 BTA Filters")
    st.markdown("---")

    selected_areas = st.multiselect(
        "📍 Areas",
        options=sorted(df_full["Area Name"].unique()),
        default=sorted(df_full["Area Name"].unique())
    )
    selected_weather = st.multiselect(
        "🌦️ Weather",
        options=sorted(df_full["Weather Conditions"].unique()),
        default=sorted(df_full["Weather Conditions"].unique())
    )
    selected_years = st.multiselect(
        "📅 Year",
        options=sorted(df_full["Year"].unique()),
        default=sorted(df_full["Year"].unique())
    )
    roadwork_filter = st.radio("🚧 Roadwork", ["All", "Active", "None"], horizontal=True)

    st.markdown("---")
    st.markdown(f"<small style='color:#555'>Model R² score: <b style='color:#ff6b35'>{model_score:.3f}</b></small>", unsafe_allow_html=True)
    st.markdown(f"<small style='color:#555'>Dataset: <b style='color:#ff6b35'>{len(df_full):,} records</b></small>", unsafe_allow_html=True)
    st.markdown(f"<small style='color:#555'>Date range: <b style='color:#ff6b35'>2022 – 2024</b></small>", unsafe_allow_html=True)

# ── Filter data ───────────────────────────────────────────────────────────────
df = df_full[
    df_full["Area Name"].isin(selected_areas) &
    df_full["Weather Conditions"].isin(selected_weather) &
    df_full["Year"].isin(selected_years)
].copy()

if roadwork_filter == "Active":
    df = df[df["Roadwork and Construction Activity"] == "Yes"]
elif roadwork_filter == "None":
    df = df[df["Roadwork and Construction Activity"] == "No"]

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">🚦 Bangalore Traffic Analyzer</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Real traffic intelligence across Bangalore\'s major corridors · 2022–2024</div>', unsafe_allow_html=True)

# ── KPI Cards ─────────────────────────────────────────────────────────────────
kpis = get_kpis(df)
c1, c2, c3, c4, c5, c6 = st.columns(6)

cards = [
    (c1, "Avg Traffic Volume", f"{kpis['avg_volume']:,}", "vehicles/day"),
    (c2, "Avg Congestion",     f"{kpis['avg_congestion']}%", "city average"),
    (c3, "Avg Speed",          f"{kpis['avg_speed']} km/h", "across all roads"),
    (c4, "Worst Area",         kpis['worst_area'], "highest volume"),
    (c5, "Worst Road",         kpis['worst_road'].split()[-1] + "...", "most congested"),
    (c6, "Total Incidents",    f"{kpis['total_incidents']:,}", "reported"),
]
for col, label, value, sub in cards:
    col.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-sub">{sub}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "📊 Overview", "🗺️ Live Map", "🔥 Deep Analysis", "🤖 Predict", "💡 Insights", "🧪 Model Testing", "📅 Year-over-Year", "⚖️ Area Compare"
])

PLOT_THEME = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font_color="#cccccc",
    xaxis=dict(gridcolor="#1e2235", linecolor="#2e3250"),
    yaxis=dict(gridcolor="#1e2235", linecolor="#2e3250"),
)

PLOT_THEME_NO_YAXIS = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font_color="#cccccc",
    xaxis=dict(gridcolor="#1e2235", linecolor="#2e3250"),
)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    # Traffic by Area
    st.markdown('<div class="section-title">🔴 Traffic Volume by Area</div>', unsafe_allow_html=True)
    wa = worst_areas(df)
    fig = px.bar(wa, x="Area Name", y="Traffic Volume",
                 color="Traffic Volume", color_continuous_scale="Reds",
                 text="Traffic Volume")
    fig.update_traces(texttemplate="%{text:,.0f}", textposition="outside")
    fig.update_layout(**PLOT_THEME, showlegend=False, height=380)
    st.plotly_chart(fig, width='stretch')

    col1, col2 = st.columns(2)

    # Top 10 roads
    with col1:
        st.markdown('<div class="section-title">🛣️ Top 10 Most Congested Roads</div>', unsafe_allow_html=True)
        wr = worst_roads(df)
        fig2 = px.bar(wr, x="Traffic Volume", y="Road/Intersection Name",
                      orientation="h", color="Traffic Volume",
                      color_continuous_scale="Oranges")
        fig2.update_layout(**PLOT_THEME_NO_YAXIS, height=380, yaxis=dict(autorange="reversed", gridcolor="#1e2235", linecolor="#2e3250"))
        st.plotly_chart(fig2, width='stretch')

    # Day of week
    with col2:
        st.markdown('<div class="section-title">📅 Traffic by Day of Week</div>', unsafe_allow_html=True)
        dp = day_patterns(df)
        fig3 = px.area(dp, x="Day", y="Traffic Volume",
                       color_discrete_sequence=["#ff6b35"])
        fig3.update_layout(**PLOT_THEME, height=380)
        fig3.update_traces(fill="tozeroy", line_color="#ff4b4b")
        st.plotly_chart(fig3, width='stretch')

    # Monthly trend
    st.markdown('<div class="section-title">📈 Monthly Traffic Trend</div>', unsafe_allow_html=True)
    mt = monthly_trend(df)
    fig4 = px.line(mt, x="Month", y="Traffic Volume", markers=True,
                   color_discrete_sequence=["#ffd700"])
    fig4.update_layout(**PLOT_THEME, height=320)
    fig4.update_traces(line_width=3, marker_size=8)
    st.plotly_chart(fig4, width='stretch')


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — LIVE MAP
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-title">🗺️ Bangalore Traffic Heatmap</div>', unsafe_allow_html=True)
    st.markdown("<small style='color:#666'>Bubble size = Traffic Volume · Color = Congestion Level</small>", unsafe_allow_html=True)

    map_df = map_data(df)

    m = folium.Map(
        location=[12.9716, 77.5946],
        zoom_start=12,
        tiles="CartoDB dark_matter"
    )

    max_vol = map_df["avg_volume"].max()
    min_vol = map_df["avg_volume"].min()

    for _, row in map_df.iterrows():
        congestion = row["avg_congestion"]
        # Color based on congestion
        if congestion >= 85:
            color = "#ff2222"
        elif congestion >= 70:
            color = "#ff8c00"
        else:
            color = "#00d084"

        radius = 15 + 35 * (row["avg_volume"] - min_vol) / (max_vol - min_vol + 1)

        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=radius,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.7,
            popup=folium.Popup(f"""
                <div style='font-family:sans-serif; min-width:180px'>
                    <b style='font-size:14px'>{row['Area Name']}</b><br>
                    <hr style='margin:4px 0'>
                    🚗 Avg Volume: <b>{int(row['avg_volume']):,}</b><br>
                    🔴 Congestion: <b>{row['avg_congestion']:.1f}%</b><br>
                    ⚡ Avg Speed:  <b>{row['avg_speed']:.1f} km/h</b><br>
                    ⚠️ Incidents: <b>{int(row['incidents'])}</b>
                </div>
            """, max_width=220),
            tooltip=f"{row['Area Name']} — {row['avg_congestion']:.0f}% congestion"
        ).add_to(m)

    st_folium(m, width="100%", height=520)

    # Area stats table
    st.markdown('<div class="section-title">📋 Area Stats Summary</div>', unsafe_allow_html=True)
    display_df = map_df[["Area Name","avg_volume","avg_congestion","avg_speed","incidents"]].copy()
    display_df.columns = ["Area","Avg Volume","Avg Congestion %","Avg Speed km/h","Total Incidents"]
    display_df = display_df.sort_values("Avg Volume", ascending=False).reset_index(drop=True)
    display_df["Avg Volume"] = display_df["Avg Volume"].apply(lambda x: f"{int(x):,}")
    display_df["Avg Congestion %"] = display_df["Avg Congestion %"].apply(lambda x: f"{x:.1f}%")
    display_df["Avg Speed km/h"] = display_df["Avg Speed km/h"].apply(lambda x: f"{x:.1f}")
    st.dataframe(display_df, width='stretch', hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — DEEP ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    col1, col2 = st.columns(2)

    # Weather impact
    with col1:
        st.markdown('<div class="section-title">🌦️ Weather Impact</div>', unsafe_allow_html=True)
        wi = weather_impact(df)
        fig5 = px.bar(wi, x="Weather Conditions", y="Congestion Level",
                      color="Congestion Level", color_continuous_scale="RdYlGn_r",
                      text="Congestion Level")
        fig5.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig5.update_layout(**PLOT_THEME, height=340)
        st.plotly_chart(fig5, width='stretch')

    # Roadwork impact
    with col2:
        st.markdown('<div class="section-title">🚧 Roadwork Impact</div>', unsafe_allow_html=True)
        ri = roadwork_impact(df)
        fig6 = px.bar(ri, x="Roadwork and Construction Activity",
                      y=["Congestion Level","Average Speed"],
                      barmode="group", color_discrete_sequence=["#ff4b4b","#00d084"])
        fig6.update_layout(**PLOT_THEME, height=340)
        st.plotly_chart(fig6, width='stretch')

    # Correlation heatmap
    st.markdown('<div class="section-title">🔥 Correlation Heatmap</div>', unsafe_allow_html=True)
    corr = correlation_data(df)
    fig7 = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns,
        y=corr.index,
        colorscale="RdBu_r",
        zmin=-1, zmax=1,
        text=corr.values,
        texttemplate="%{text:.2f}",
        textfont_size=11,
        hoverongaps=False
    ))
    fig7.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#cccccc",
        height=480,
        margin=dict(l=180, r=20, t=20, b=180)
    )
    st.plotly_chart(fig7, width='stretch')

    # Speed vs Volume scatter
    st.markdown('<div class="section-title">⚡ Speed vs Volume by Area</div>', unsafe_allow_html=True)
    fig8 = px.scatter(df, x="Traffic Volume", y="Average Speed",
                      color="Area Name", size="Congestion Level",
                      hover_data=["Road/Intersection Name","Weather Conditions"],
                      color_discrete_sequence=px.colors.qualitative.Bold,
                      opacity=0.7)
    fig8.update_layout(**PLOT_THEME, height=420)
    st.plotly_chart(fig8, width='stretch')


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — PREDICT
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-title">🤖 Congestion Predictor</div>', unsafe_allow_html=True)
    st.markdown(f"<small style='color:#666'>Powered by Random Forest · Model accuracy (R²): <b style='color:#ff6b35'>{model_score:.3f}</b></small>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown('<div class="predict-card">', unsafe_allow_html=True)
        p_area    = st.selectbox("📍 Area", sorted(df_full["Area Name"].unique()))
        p_road    = st.selectbox("🛣️ Road", sorted(df_full[df_full["Area Name"]==p_area]["Road/Intersection Name"].unique()))
        p_weather = st.selectbox("🌦️ Weather", sorted(df_full["Weather Conditions"].unique()))
        p_day     = st.selectbox("📅 Day", ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"])
        p_month   = st.selectbox("🗓️ Month", ["January","February","March","April","May","June","July","August","September","October","November","December"])
        p_roadwork= st.toggle("🚧 Roadwork Active?", value=False)
        p_volume  = st.slider("🚗 Expected Traffic Volume", 4000, 75000, 29000, 500)
        p_speed   = st.slider("⚡ Expected Avg Speed (km/h)", 10, 80, 40, 1)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        day_map   = {"Monday":0,"Tuesday":1,"Wednesday":2,"Thursday":3,"Friday":4,"Saturday":5,"Sunday":6}
        month_map = {"January":1,"February":2,"March":3,"April":4,"May":5,"June":6,
                     "July":7,"August":8,"September":9,"October":10,"November":11,"December":12}

        prediction = predict_congestion(
            model, le_area, le_weather, le_road, features,
            p_area, p_weather, p_road,
            day_map[p_day], month_map[p_month],
            p_roadwork, p_volume, p_speed
        )

        if prediction >= 80:
            level, color, emoji, advice = "HIGH", "#ff4b4b", "🔴", "Expect severe delays. Consider alternate routes or off-peak travel."
        elif prediction >= 50:
            level, color, emoji, advice = "MEDIUM", "#ff8c00", "🟠", "Moderate traffic expected. Allow extra travel time."
        else:
            level, color, emoji, advice = "LOW", "#00d084", "🟢", "Roads are relatively clear. Good time to travel."

        st.markdown(f"""
        <div style='background:linear-gradient(135deg,#1a1d27,#22263a);border:1px solid #2e3250;
                    border-radius:16px;padding:2rem;text-align:center;margin-top:1rem'>
            <div style='color:#888;font-size:0.9rem;text-transform:uppercase;letter-spacing:2px'>
                Predicted Congestion Level
            </div>
            <div style='color:{color};font-size:4rem;font-weight:800;margin:0.5rem 0'>
                {prediction}%
            </div>
            <div style='color:{color};font-size:1.4rem;font-weight:600;margin-bottom:1rem'>
                {emoji} {level} CONGESTION
            </div>
            <hr style='border-color:#2e3250'>
            <div style='color:#aaa;font-size:0.95rem;margin-top:1rem'>{advice}</div>
            <br>
            <div style='display:flex;justify-content:space-around;margin-top:0.5rem'>
                <div>
                    <div style='color:#555;font-size:0.75rem'>AREA</div>
                    <div style='color:#fff;font-weight:600'>{p_area}</div>
                </div>
                <div>
                    <div style='color:#555;font-size:0.75rem'>DAY</div>
                    <div style='color:#fff;font-weight:600'>{p_day}</div>
                </div>
                <div>
                    <div style='color:#555;font-size:0.75rem'>WEATHER</div>
                    <div style='color:#fff;font-weight:600'>{p_weather}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Gauge chart
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prediction,
            domain={"x":[0,1],"y":[0,1]},
            gauge={
                "axis":  {"range":[0,100], "tickcolor":"#555"},
                "bar":   {"color": color},
                "bgcolor": "#1a1d27",
                "steps": [
                    {"range":[0,50],   "color":"#0d2b1a"},
                    {"range":[50,80],  "color":"#2b1f0a"},
                    {"range":[80,100], "color":"#2b0a0a"},
                ],
                "threshold": {"line":{"color":"white","width":3},"thickness":0.8,"value":prediction}
            },
            number={"suffix":"%","font":{"color":color,"size":32}}
        ))
        fig_gauge.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="#ccc",
            height=240,
            margin=dict(t=30,b=10,l=30,r=30)
        )
        st.plotly_chart(fig_gauge, width='stretch')


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — INSIGHTS
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown('<div class="section-title">💡 Auto-Generated Insights</div>', unsafe_allow_html=True)
    st.markdown("<small style='color:#666'>Dynamically generated from the filtered dataset</small><br><br>", unsafe_allow_html=True)

    insights = generate_insights(df)
    for insight in insights:
        st.markdown(f'<div class="insight-card">{insight}</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Feature importance
    st.markdown('<div class="section-title">🧠 What Drives Congestion? (ML Feature Importance)</div>', unsafe_allow_html=True)
    feat_names = ["Area","Weather","Road","Day of Week","Month","Roadwork","Traffic Volume","Avg Speed"]
    importances = model.feature_importances_
    fi_df = pd.DataFrame({"Feature": feat_names, "Importance": importances})\
              .sort_values("Importance", ascending=True)
    fig9 = px.bar(fi_df, x="Importance", y="Feature", orientation="h",
                  color="Importance", color_continuous_scale="Oranges")
    fig9.update_layout(**PLOT_THEME, height=360)
    st.plotly_chart(fig9, width='stretch')

    # Congestion distribution
    st.markdown('<div class="section-title">📊 Congestion Level Distribution</div>', unsafe_allow_html=True)
    fig10 = px.histogram(df, x="Congestion Level", nbins=40,
                         color_discrete_sequence=["#ff4b4b"])
    fig10.update_layout(**PLOT_THEME, height=320)
    st.plotly_chart(fig10, width='stretch')

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<hr style='border-color:#1e2235;margin-top:3rem'>
<div style='text-align:center;color:#333;font-size:0.8rem;padding:1rem'>
    Bangalore Traffic Analyzer · Built with Streamlit & Python · Data: 2022–2024
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — MODEL TESTING
# ══════════════════════════════════════════════════════════════════════════════
with tab6:
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import mean_absolute_error
    from sklearn.preprocessing import LabelEncoder
    import numpy as np

    st.markdown('<div class="section-title">🧪 Model Accuracy Testing</div>', unsafe_allow_html=True)
    st.markdown("<small style='color:#666'>Testing only on REAL data (2022–2024 Kaggle dataset) — not synthetic</small><br>", unsafe_allow_html=True)

    df_real = df_full[df_full["Year"] <= 2024].copy()
    le_a2 = LabelEncoder().fit(df_full["Area Name"])
    le_w2 = LabelEncoder().fit(df_full["Weather Conditions"])
    le_r2 = LabelEncoder().fit(df_full["Road/Intersection Name"])
    df_real["area_enc"]    = le_a2.transform(df_real["Area Name"])
    df_real["weather_enc"] = le_w2.transform(df_real["Weather Conditions"])
    df_real["road_enc"]    = le_r2.transform(df_real["Road/Intersection Name"])
    X_real = df_real[features]
    y_real = df_real["Congestion Level"]
    preds  = model.predict(X_real)
    mae    = mean_absolute_error(y_real, preds)
    r2_real= r2_score(y_real, preds)

    c1, c2, c3, c4 = st.columns(4)
    for col, label, value, sub in [
        (c1, "R² Score",         f"{r2_real:.3f}", "1.0 = perfect"),
        (c2, "Mean Abs Error",   f"{mae:.2f}%",    "avg prediction error"),
        (c3, "Real Rows Tested", f"{len(df_real):,}", "2022–2024 only"),
        (c4, "Avg Error Band",   f"±{mae:.1f}%",   "typical deviation"),
    ]:
        col.markdown(f'<div class="kpi-card"><div class="kpi-label">{label}</div><div class="kpi-value">{value}</div><div class="kpi-sub">{sub}</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Predicted vs Actual scatter
    st.markdown('<div class="section-title">🎯 Predicted vs Actual Congestion</div>', unsafe_allow_html=True)
    st.markdown("<small style='color:#666'>Perfect model = all dots on the white diagonal line</small>", unsafe_allow_html=True)
    sidx = np.random.choice(len(df_real), size=min(500, len(df_real)), replace=False)
    scatter_df = pd.DataFrame({
        "Actual":    y_real.iloc[sidx].values,
        "Predicted": preds[sidx],
        "Area":      df_real["Area Name"].iloc[sidx].values,
        "Error":     abs(y_real.iloc[sidx].values - preds[sidx])
    })
    fig_sc = px.scatter(scatter_df, x="Actual", y="Predicted", color="Area",
                        size="Error", opacity=0.7,
                        color_discrete_sequence=px.colors.qualitative.Bold)
    fig_sc.add_shape(type="line", x0=0, y0=0, x1=100, y1=100,
                     line=dict(color="#ffffff", width=2, dash="dash"))
    fig_sc.update_layout(**PLOT_THEME, height=450,
                          xaxis_title="Actual Congestion %",
                          yaxis_title="Predicted Congestion %")
    st.plotly_chart(fig_sc, width='stretch')

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-title">📊 Error Distribution</div>', unsafe_allow_html=True)
        err_df = pd.DataFrame({"Absolute Error (%)": abs(y_real.values - preds)})
        fig_err = px.histogram(err_df, x="Absolute Error (%)", nbins=50,
                               color_discrete_sequence=["#ff6b35"])
        fig_err.add_vline(x=mae, line_dash="dash", line_color="white",
                          annotation_text=f"Mean: {mae:.1f}%")
        fig_err.update_layout(**PLOT_THEME, height=320)
        st.plotly_chart(fig_err, width='stretch')

    with col2:
        st.markdown('<div class="section-title">📋 20 Random Real Predictions</div>', unsafe_allow_html=True)
        s20   = df_real.sample(20, random_state=99)
        p20   = model.predict(s20[features])
        cmp_df = pd.DataFrame({
            "Area":        s20["Area Name"].values,
            "Actual %":    s20["Congestion Level"].round(1).values,
            "Predicted %": p20.round(1),
            "Error %":     abs(s20["Congestion Level"].values - p20).round(1),
        })
        st.dataframe(cmp_df, hide_index=True, height=340)

    # Cross validation
    st.markdown('<div class="section-title">🔄 5-Fold Cross Validation</div>', unsafe_allow_html=True)
    st.markdown("<small style='color:#666'>Tests across 5 different splits — more reliable than a single score</small>", unsafe_allow_html=True)
    with st.spinner("Running cross validation... (takes ~30 seconds)"):
        cv_scores = cross_val_score(model, X_real, y_real, cv=5, scoring='r2')
    cv_df = pd.DataFrame({
        "Fold":     [f"Fold {i+1}" for i in range(5)],
        "R² Score": cv_scores.round(3),
        "Quality":  ["✅ Excellent" if s > 0.9 else "🟠 Good" if s > 0.8 else "🔴 Weak" for s in cv_scores]
    })
    fig_cv = px.bar(cv_df, x="Fold", y="R² Score", color="R² Score",
                    color_continuous_scale="RdYlGn", text="R² Score", range_y=[0,1])
    fig_cv.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    fig_cv.add_hline(y=cv_scores.mean(), line_dash="dash", line_color="white",
                     annotation_text=f"Mean: {cv_scores.mean():.3f}")
    fig_cv.update_layout(**PLOT_THEME, height=360)
    st.plotly_chart(fig_cv, width='stretch')
    st.dataframe(cv_df, hide_index=True, use_container_width=False)
    st.markdown(f"""
    <div class="insight-card" style="margin-top:1rem">
        📌 <b>Summary:</b> Average R² across 5 splits = <b style="color:#ff6b35">{cv_scores.mean():.3f}</b>
        (±{cv_scores.std():.3f}). Mean absolute error = <b style="color:#ff6b35">{mae:.1f}%</b>.
        Predictions are typically within {mae:.1f} percentage points of real congestion.
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 7 — YEAR-OVER-YEAR
# ══════════════════════════════════════════════════════════════════════════════
with tab7:
    st.markdown('<div class="section-title">📅 Year-over-Year Traffic Trends</div>', unsafe_allow_html=True)
    st.markdown("<small style='color:#666'>Has Bangalore traffic gotten worse? Real data 2022–2024 · Projected 2025–2026</small><br>", unsafe_allow_html=True)

    yoy_df = df_full.copy()
    yoy_df["Year"] = yoy_df["Year"].astype(str)

    # ── Overall YoY line ──────────────────────────────────────────────────────
    yoy_monthly = yoy_df.groupby(["Year","MonthNum","Month"])[
        ["Traffic Volume","Congestion Level","Average Speed"]
    ].mean().reset_index()
    yoy_monthly = yoy_monthly.sort_values(["Year","MonthNum"])

    st.markdown('<div class="section-title">📈 Monthly Traffic Volume — All Years</div>', unsafe_allow_html=True)
    fig_yoy1 = px.line(
        yoy_monthly, x="Month", y="Traffic Volume",
        color="Year", markers=True,
        color_discrete_sequence=["#4cc9f0","#7bed9f","#ffd700","#ff6b35","#ff4b4b"],
        category_orders={"Month": ["January","February","March","April","May","June",
                                    "July","August","September","October","November","December"]}
    )
    fig_yoy1.update_traces(line_width=2.5, marker_size=7)
    fig_yoy1.update_layout(**PLOT_THEME, height=400)
    st.plotly_chart(fig_yoy1, width='stretch')

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-title">🔴 Avg Congestion per Year</div>', unsafe_allow_html=True)
        yoy_annual = yoy_df.groupby("Year")[
            ["Traffic Volume","Congestion Level","Average Speed","Incident Reports"]
        ].mean().reset_index()
        fig_yoy2 = px.bar(
            yoy_annual, x="Year", y="Congestion Level",
            color="Congestion Level", color_continuous_scale="RdYlGn_r",
            text="Congestion Level"
        )
        fig_yoy2.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig_yoy2.update_layout(**PLOT_THEME, height=350)
        st.plotly_chart(fig_yoy2, width='stretch')

    with col2:
        st.markdown('<div class="section-title">⚡ Avg Speed per Year</div>', unsafe_allow_html=True)
        fig_yoy3 = px.bar(
            yoy_annual, x="Year", y="Average Speed",
            color="Average Speed", color_continuous_scale="RdYlGn",
            text="Average Speed"
        )
        fig_yoy3.update_traces(texttemplate="%{text:.1f} km/h", textposition="outside")
        fig_yoy3.update_layout(**PLOT_THEME, height=350)
        st.plotly_chart(fig_yoy3, width='stretch')

    # ── Per area YoY ──────────────────────────────────────────────────────────
    st.markdown('<div class="section-title">🏙️ Congestion Growth by Area (2022 → 2026)</div>', unsafe_allow_html=True)
    area_yoy = yoy_df.groupby(["Year","Area Name"])["Congestion Level"].mean().reset_index()
    fig_yoy4 = px.line(
        area_yoy, x="Year", y="Congestion Level",
        color="Area Name", markers=True,
        color_discrete_sequence=px.colors.qualitative.Bold
    )
    fig_yoy4.update_traces(line_width=2, marker_size=8)
    fig_yoy4.update_layout(**PLOT_THEME, height=420)
    st.plotly_chart(fig_yoy4, width='stretch')

    # ── YoY summary table ─────────────────────────────────────────────────────
    st.markdown('<div class="section-title">📋 Annual Summary Table</div>', unsafe_allow_html=True)
    summary = yoy_df.groupby("Year").agg(
        Avg_Volume      =("Traffic Volume",   "mean"),
        Avg_Congestion  =("Congestion Level", "mean"),
        Avg_Speed       =("Average Speed",    "mean"),
        Total_Incidents =("Incident Reports", "sum"),
    ).reset_index()
    summary.columns = ["Year","Avg Volume","Avg Congestion %","Avg Speed km/h","Total Incidents"]
    summary["Avg Volume"]       = summary["Avg Volume"].apply(lambda x: f"{int(x):,}")
    summary["Avg Congestion %"] = summary["Avg Congestion %"].apply(lambda x: f"{x:.1f}%")
    summary["Avg Speed km/h"]   = summary["Avg Speed km/h"].apply(lambda x: f"{x:.1f}")
    summary["Total Incidents"]  = summary["Total Incidents"].apply(lambda x: f"{int(x):,}")
    st.dataframe(summary, hide_index=True, use_container_width=False)

    # ── Key insight ───────────────────────────────────────────────────────────
    real_2022 = yoy_df[yoy_df["Year"]=="2022"]["Congestion Level"].mean()
    real_2024 = yoy_df[yoy_df["Year"]=="2024"]["Congestion Level"].mean()
    delta     = real_2024 - real_2022
    direction = "increased" if delta > 0 else "decreased"
    st.markdown(f"""
    <div class="insight-card" style="margin-top:1rem">
        📌 <b>Key Finding:</b> Bangalore's average congestion <b>{direction}</b> by
        <b style="color:#ff6b35">{abs(delta):.1f} percentage points</b> from 2022 to 2024
        (real data). {"Traffic is getting worse year on year." if delta > 0 else "Traffic conditions have slightly improved."}
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 8 — AREA COMPARE
# ══════════════════════════════════════════════════════════════════════════════
with tab8:
    st.markdown('<div class="section-title">⚖️ Area Comparison Tool</div>', unsafe_allow_html=True)
    st.markdown("<small style='color:#666'>Pick any two areas and compare them head-to-head</small><br>", unsafe_allow_html=True)

    all_areas = sorted(df_full["Area Name"].unique())

    col1, col2 = st.columns(2)
    with col1:
        area_a = st.selectbox("🔴 Area A", all_areas, index=all_areas.index("Koramangala"))
    with col2:
        area_b = st.selectbox("🔵 Area B", all_areas, index=all_areas.index("Electronic City"))

    df_a = df_full[df_full["Area Name"] == area_a]
    df_b = df_full[df_full["Area Name"] == area_b]

    # ── Head-to-head KPI cards ────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    metrics_compare = [
        ("Avg Traffic Volume",    int(df_a["Traffic Volume"].mean()),   int(df_b["Traffic Volume"].mean()),   "{:,}"),
        ("Avg Congestion %",      df_a["Congestion Level"].mean(),      df_b["Congestion Level"].mean(),      "{:.1f}%"),
        ("Avg Speed km/h",        df_a["Average Speed"].mean(),         df_b["Average Speed"].mean(),         "{:.1f}"),
        ("Total Incidents",       int(df_a["Incident Reports"].sum()),  int(df_b["Incident Reports"].sum()),  "{:,}"),
        ("Avg Road Capacity %",   df_a["Road Capacity Utilization"].mean(), df_b["Road Capacity Utilization"].mean(), "{:.1f}%"),
    ]

    header_cols = st.columns([2,1,1])
    header_cols[0].markdown(f"<div style='color:#888;font-size:0.8rem;text-transform:uppercase'>Metric</div>", unsafe_allow_html=True)
    header_cols[1].markdown(f"<div style='color:#ff4b4b;font-size:0.9rem;font-weight:700;text-align:center'>🔴 {area_a}</div>", unsafe_allow_html=True)
    header_cols[2].markdown(f"<div style='color:#4cc9f0;font-size:0.9rem;font-weight:700;text-align:center'>🔵 {area_b}</div>", unsafe_allow_html=True)

    st.markdown("<hr style='border-color:#1e2235;margin:0.3rem 0'>", unsafe_allow_html=True)

    for label, val_a, val_b, fmt in metrics_compare:
        row = st.columns([2,1,1])
        row[0].markdown(f"<div style='color:#888;padding:0.4rem 0'>{label}</div>", unsafe_allow_html=True)
        color_a = "#ff4b4b" if val_a >= val_b else "#00d084"
        color_b = "#ff4b4b" if val_b >= val_a else "#00d084"
        # For speed, higher is better — flip colors
        if "Speed" in label:
            color_a = "#00d084" if val_a >= val_b else "#ff4b4b"
            color_b = "#00d084" if val_b >= val_a else "#ff4b4b"
        row[1].markdown(f"<div style='color:{color_a};font-weight:700;text-align:center;padding:0.4rem 0'>{fmt.format(val_a)}</div>", unsafe_allow_html=True)
        row[2].markdown(f"<div style='color:{color_b};font-weight:700;text-align:center;padding:0.4rem 0'>{fmt.format(val_b)}</div>", unsafe_allow_html=True)
        st.markdown("<hr style='border-color:#1e2235;margin:0'>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Side by side charts ───────────────────────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f'<div class="section-title">📅 {area_a} — Day Patterns</div>', unsafe_allow_html=True)
        order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
        dp_a = df_a.groupby("Day")["Congestion Level"].mean().reindex(order).reset_index()
        fig_ca = px.bar(dp_a, x="Day", y="Congestion Level",
                        color="Congestion Level", color_continuous_scale="Reds")
        fig_ca.update_layout(**PLOT_THEME, height=300, showlegend=False)
        st.plotly_chart(fig_ca, width='stretch')

    with col2:
        st.markdown(f'<div class="section-title">📅 {area_b} — Day Patterns</div>', unsafe_allow_html=True)
        dp_b = df_b.groupby("Day")["Congestion Level"].mean().reindex(order).reset_index()
        fig_cb = px.bar(dp_b, x="Day", y="Congestion Level",
                        color="Congestion Level", color_continuous_scale="Blues")
        fig_cb.update_layout(**PLOT_THEME, height=300, showlegend=False)
        st.plotly_chart(fig_cb, width='stretch')

    # ── Weather comparison ────────────────────────────────────────────────────
    st.markdown('<div class="section-title">🌦️ Weather Impact Comparison</div>', unsafe_allow_html=True)
    w_a = df_a.groupby("Weather Conditions")["Congestion Level"].mean().reset_index()
    w_a["Area"] = area_a
    w_b = df_b.groupby("Weather Conditions")["Congestion Level"].mean().reset_index()
    w_b["Area"] = area_b
    w_combined = pd.concat([w_a, w_b])
    fig_wc = px.bar(w_combined, x="Weather Conditions", y="Congestion Level",
                    color="Area", barmode="group",
                    color_discrete_map={area_a: "#ff4b4b", area_b: "#4cc9f0"})
    fig_wc.update_layout(**PLOT_THEME, height=350)
    st.plotly_chart(fig_wc, width='stretch')

    # ── YoY comparison ────────────────────────────────────────────────────────
    st.markdown('<div class="section-title">📈 Year-over-Year Congestion Comparison</div>', unsafe_allow_html=True)
    yoy_a = df_a.groupby("Year")["Congestion Level"].mean().reset_index()
    yoy_a["Area"] = area_a
    yoy_b = df_b.groupby("Year")["Congestion Level"].mean().reset_index()
    yoy_b["Area"] = area_b
    yoy_combined = pd.concat([yoy_a, yoy_b])
    yoy_combined["Year"] = yoy_combined["Year"].astype(str)
    fig_yoy_c = px.line(yoy_combined, x="Year", y="Congestion Level",
                        color="Area", markers=True,
                        color_discrete_map={area_a: "#ff4b4b", area_b: "#4cc9f0"})
    fig_yoy_c.update_traces(line_width=3, marker_size=10)
    fig_yoy_c.update_layout(**PLOT_THEME, height=350)
    st.plotly_chart(fig_yoy_c, width='stretch')

    # ── Verdict ───────────────────────────────────────────────────────────────
    cong_a = df_a["Congestion Level"].mean()
    cong_b = df_b["Congestion Level"].mean()
    spd_a  = df_a["Average Speed"].mean()
    spd_b  = df_b["Average Speed"].mean()
    winner = area_b if cong_b < cong_a else area_a
    st.markdown(f"""
    <div class="insight-card" style="margin-top:1rem">
        ⚖️ <b>Verdict:</b> <b style="color:#ff6b35">{winner}</b> is the better area to drive in —
        {f"{area_a} has {cong_a:.1f}% avg congestion vs {area_b}'s {cong_b:.1f}%" }.
        Drivers in {area_a} average <b>{spd_a:.1f} km/h</b> vs <b>{spd_b:.1f} km/h</b> in {area_b}.
    </div>
    """, unsafe_allow_html=True)