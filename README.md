🚦 Bangalore Traffic Intelligence Dashboard

An end-to-end traffic analysis and prediction platform for 13 major Bangalore corridors — built with Python, Machine Learning, and Streamlit.

📌 What This Project Does
Most traffic tools just show you current conditions. This dashboard goes further — it analyzes historical patterns, predicts future congestion, and lets you compare areas head-to-head across 4 years of data.
Built as a full data pipeline: raw CSV → cleaning → analysis → ML model → deployed dashboard.

🖥️ Live Demo
👉 bangalore-traffic-analyzer.streamlit.app

📊 Dashboard Features (8 Tabs)
TabWhat It Shows📊 OverviewTraffic volume by area, top 10 congested roads, day-of-week patterns, monthly trends🗺️ Live MapInteractive Folium dark map with color-coded congestion bubbles per area🔥 Deep AnalysisCorrelation heatmap, weather impact, roadwork analysis, speed vs volume scatter🤖 PredictML-powered congestion predictor with gauge chart and travel advice💡 Insights6 auto-generated insights from real data + ML feature importance chart🧪 Model TestingPredicted vs actual scatter, error distribution, 5-fold cross validation📅 Year-over-YearHas Bangalore traffic gotten worse? Full 2022–2026 trend analysis⚖️ Area ComparePick any 2 areas and compare congestion, speed, incidents head-to-head

🤖 Machine Learning Model

Algorithm: Random Forest Regressor (100 estimators)
Target: Congestion Level (%)
Features: Area, Road, Weather, Day of Week, Month, Roadwork status, Traffic Volume, Avg Speed
R² Score: 0.94 on test set
Mean Absolute Error: 1.8 percentage points
Cross Validation: 5-fold CV mean R² = 0.846 (tested on real data only)


📦 Dataset
PropertyDetailsTotal Records25,000+Areas Covered13 major Bangalore corridorsDate RangeJanuary 2022 – April 2026Real Data2022–2024 (Kaggle source)Extended Data2024–2026 (statistically generated from real distributions)
Areas included:
Indiranagar · Whitefield · Koramangala · M.G. Road · Jayanagar · Hebbal · Yeshwanthpur · Electronic City · RR Nagar · JP Nagar · Kanakapura Road · Girinagar · Mysore Road

🛠️ Tech Stack
LayerToolsLanguagePython 3.12Data ProcessingPandas, NumPyMachine LearningScikit-learn (Random Forest)VisualizationPlotly, FoliumDashboardStreamlitDeploymentStreamlit CloudVersion ControlGit, GitHub

🚀 Run Locally
1. Clone the repository
bashgit clone https://github.com/Samarthsj1013/bangalore-traffic-analyzer.git
cd bangalore-traffic-analyzer
2. Install dependencies
bashpip install -r requirements.txt
3. Run the app
bashstreamlit run app.py
4. Open in browser
http://localhost:8501

📁 Project Structure
bangalore-traffic-analyzer/
│
├── data/
│   └── Banglore_traffic_Dataset.csv   # Extended dataset (2022–2026)
│
├── app.py                              # Main Streamlit dashboard (8 tabs)
├── analysis.py                         # Data processing & ML pipeline
├── requirements.txt                    # Python dependencies
└── README.md

🔍 Key Findings

Koramangala is consistently the most congested area — 1.5x above city average
Sony World Junction is the single most congested road in the dataset
Wednesday is the busiest day; Friday is the quietest — a ~10% swing
Rain causes slightly higher traffic volume than clear weather
Traffic Volume is the #1 predictor of congestion (per ML feature importance)
Average speeds are declining year-over-year indicating worsening conditions


👤 Author
Samarth Jayant

📧 samarthsj1013@gmail.com
🐙 github.com/Samarthsj1013
🎓 B.E. Information Science & Engineering, Global Academy of Technology, Bangalore


📄 Data Source
Original dataset: Bangalore City Traffic Dataset — Kaggle
Extended with statistically generated data for additional areas and 2024–2026 period based on real distributions.

Built as part of a Data Analysis & Engineering portfolio project.