import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import pickle
import os

# ── Area coordinates for Folium map ──────────────────────────────────────────
AREA_COORDS = {
    "Indiranagar":     (12.9784, 77.6408),
    "Whitefield":      (12.9698, 77.7499),
    "Koramangala":     (12.9352, 77.6245),
    "M.G. Road":       (12.9757, 77.6011),
    "Jayanagar":       (12.9250, 77.5938),
    "Hebbal":          (13.0450, 77.5970),
    "Yeshwanthpur":    (13.0246, 77.5530),
    "Electronic City": (12.8399, 77.6770),
    "RR Nagar":        (12.9230, 77.5190),
    "JP Nagar":        (12.9102, 77.5850),
    "Kanakapura Road": (12.8950, 77.5750),
    "Girinagar":       (12.9390, 77.5480),
    "Mysore Road":     (12.9540, 77.4980),
}

def load_data(path="data/Banglore_traffic_Dataset.csv"):
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    df["Month"]      = df["Date"].dt.month_name()
    df["MonthNum"]   = df["Date"].dt.month
    df["Day"]        = df["Date"].dt.day_name()
    df["DayNum"]     = df["Date"].dt.dayofweek
    df["Year"]       = df["Date"].dt.year
    df["Roadwork_bin"] = (df["Roadwork and Construction Activity"] == "Yes").astype(int)
    return df

# ── KPIs ─────────────────────────────────────────────────────────────────────
def get_kpis(df):
    return {
        "avg_volume":     int(df["Traffic Volume"].mean()),
        "avg_congestion": round(df["Congestion Level"].mean(), 1),
        "avg_speed":      round(df["Average Speed"].mean(), 1),
        "worst_area":     df.groupby("Area Name")["Traffic Volume"].mean().idxmax(),
        "worst_road":     df.groupby("Road/Intersection Name")["Traffic Volume"].mean().idxmax(),
        "total_incidents":int(df["Incident Reports"].sum()),
    }

# ── Chart data ────────────────────────────────────────────────────────────────
def worst_areas(df):
    return df.groupby("Area Name")["Traffic Volume"].mean().sort_values(ascending=False).reset_index()

def worst_roads(df):
    return df.groupby("Road/Intersection Name")["Traffic Volume"].mean()\
             .sort_values(ascending=False).head(10).reset_index()

def congestion_by_area(df):
    return df.groupby("Area Name")["Congestion Level"].mean()\
             .sort_values(ascending=False).reset_index()

def weather_impact(df):
    return df.groupby("Weather Conditions")[["Traffic Volume","Congestion Level","Average Speed"]]\
             .mean().reset_index()

def day_patterns(df):
    order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    return df.groupby("Day")["Traffic Volume"].mean().reindex(order).reset_index()

def monthly_trend(df):
    month_order = ["January","February","March","April","May","June",
                   "July","August","September","October","November","December"]
    return df.groupby("Month")["Traffic Volume"].mean()\
             .reindex(month_order).dropna().reset_index()

def roadwork_impact(df):
    return df.groupby("Roadwork and Construction Activity")\
             [["Congestion Level","Traffic Volume","Average Speed"]].mean().reset_index()

def correlation_data(df):
    cols = ["Traffic Volume","Average Speed","Congestion Level",
            "Road Capacity Utilization","Incident Reports",
            "Public Transport Usage","Pedestrian and Cyclist Count",
            "Environmental Impact"]
    return df[cols].corr().round(2)

def map_data(df):
    area_stats = df.groupby("Area Name").agg(
        avg_volume   =("Traffic Volume",   "mean"),
        avg_congestion=("Congestion Level","mean"),
        avg_speed    =("Average Speed",    "mean"),
        incidents    =("Incident Reports", "sum"),
    ).reset_index()
    area_stats["lat"] = area_stats["Area Name"].map(lambda x: AREA_COORDS.get(x,(12.97,77.59))[0])
    area_stats["lon"] = area_stats["Area Name"].map(lambda x: AREA_COORDS.get(x,(12.97,77.59))[1])
    return area_stats

# ── Auto-generated insights ───────────────────────────────────────────────────
def generate_insights(df):
    insights = []
    area_vol  = df.groupby("Area Name")["Traffic Volume"].mean()
    city_avg  = area_vol.mean()
    worst     = area_vol.idxmax()
    ratio     = area_vol.max() / city_avg

    insights.append(f"🔴 **{worst}** has {ratio:.1f}x higher traffic than the city average.")

    day_vol = df.groupby("Day")["Traffic Volume"].mean()
    busiest = day_vol.idxmax()
    quietest= day_vol.idxmin()
    insights.append(f"📅 **{busiest}** is the busiest day; **{quietest}** is the quietest — a {((day_vol.max()-day_vol.min())/day_vol.min()*100):.0f}% swing.")

    rain_vol = df[df["Weather Conditions"]=="Rain"]["Traffic Volume"].mean()
    clear_vol= df[df["Weather Conditions"]=="Clear"]["Traffic Volume"].mean()
    diff     = ((rain_vol - clear_vol)/clear_vol*100)
    direction= "higher" if diff > 0 else "lower"
    insights.append(f"🌧️ Rain causes **{abs(diff):.1f}% {direction}** traffic volume compared to clear weather.")

    rw_yes = df[df["Roadwork and Construction Activity"]=="Yes"]["Congestion Level"].mean()
    rw_no  = df[df["Roadwork and Construction Activity"]=="No"]["Congestion Level"].mean()
    insights.append(f"🚧 Roadwork increases congestion by **{(rw_yes-rw_no):.1f} points** on average.")

    road_vol = df.groupby("Road/Intersection Name")["Traffic Volume"].mean()
    insights.append(f"🛣️ **{road_vol.idxmax()}** is the single most congested road in the dataset.")

    speed_area = df.groupby("Area Name")["Average Speed"].mean()
    insights.append(f"🚗 Drivers move fastest in **{speed_area.idxmax()}** ({speed_area.max():.1f} km/h avg) and slowest in **{speed_area.idxmin()}** ({speed_area.min():.1f} km/h avg).")

    return insights

# ── ML Model ──────────────────────────────────────────────────────────────────
def train_model(df):
    le_area    = LabelEncoder()
    le_weather = LabelEncoder()
    le_road    = LabelEncoder()

    df2 = df.copy()
    df2["area_enc"]    = le_area.fit_transform(df2["Area Name"])
    df2["weather_enc"] = le_weather.fit_transform(df2["Weather Conditions"])
    df2["road_enc"]    = le_road.fit_transform(df2["Road/Intersection Name"])

    features = ["area_enc","weather_enc","road_enc","DayNum",
                "MonthNum","Roadwork_bin","Traffic Volume","Average Speed"]
    target   = "Congestion Level"

    X = df2[features]
    y = df2[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    score = r2_score(y_test, model.predict(X_test))

    return model, le_area, le_weather, le_road, score, features

def predict_congestion(model, le_area, le_weather, le_road, features,
                        area, weather, road, day_num, month_num,
                        roadwork, traffic_vol, avg_speed):
    row = {
        "area_enc":    le_area.transform([area])[0],
        "weather_enc": le_weather.transform([weather])[0],
        "road_enc":    le_road.transform([road])[0],
        "DayNum":      day_num,
        "MonthNum":    month_num,
        "Roadwork_bin":int(roadwork),
        "Traffic Volume": traffic_vol,
        "Average Speed":  avg_speed,
    }
    X = pd.DataFrame([row])[features]
    return round(float(model.predict(X)[0]), 1)