# =========================================================
# Zomato Last-Mile Delivery Intelligence
# Professional Analytics & Predictive Dashboard
# =========================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------
st.set_page_config(
    page_title="Zomato Last-Mile Delivery Intelligence",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# CUSTOM CSS - Clean Professional Light Theme
# ---------------------------------------------------------
st.markdown("""
<style>
    .stApp {
        background-color: #F1F5F9;
        color: #0F172A;
    }

    h1, h2, h3, h4, h5, h6 {
        color: #0F172A !important;
        font-weight: 700 !important;
    }

    p, label, span, div {
        color: #0F172A !important;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #FFFFFF;
        border-right: 1px solid #E2E8F0;
    }
    section[data-testid="stSidebar"] * {
        color: #0F172A !important;
    }

    /* Metric Cards */
    div[data-testid="stMetric"] {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-left: 5px solid #0D9488;
        border-radius: 12px;
        padding: 18px 16px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }
    div[data-testid="stMetric"] label {
        color: #64748B !important;
        font-size: 0.87rem !important;
        font-weight: 500 !important;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: #0F172A !important;
        font-size: 1.55rem !important;
        font-weight: 700 !important;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab"] {
        background-color: #FFFFFF;
        color: #0F172A !important;
        border: 1px solid #E2E8F0;
        border-radius: 8px 8px 0 0;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background-color: #0D9488 !important;
        color: white !important;
        border-color: #0D9488 !important;
    }

    /* Buttons */
    .stDownloadButton button {
        background-color: #0D9488 !important;
        color: white !important;
        border-radius: 8px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# PATHS
# ---------------------------------------------------------
BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "data" / "zomato_cleaned.csv"
DOCS_PATH = BASE_DIR / "documentation" / "System_Documentation.md"

# ---------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df["Order_Date"] = pd.to_datetime(df["Order_Date"])
    return df

df = load_data()

# ---------------------------------------------------------
# SIDEBAR - FILTERS + HELP
# ---------------------------------------------------------
st.sidebar.markdown("## Filters")
st.sidebar.markdown("---")

with st.sidebar.expander("Help & Guide", expanded=False):
    st.markdown("""
**How to use this dashboard**

- Use the filters to focus on specific cities, weather, traffic, etc.
- All charts and KPIs update automatically.
- The Prediction model is trained on the full dataset for stability.

**Field Guide**
- **City** → Delivery location  
- **Weather** → Weather condition during the order  
- **Traffic Density** → Level of road congestion  
- **Vehicle Type** → Type of vehicle used  
- **Festival** → Whether the day was a festival  
- **Distance** → Kilometers between restaurant and customer  
- **Delivery Time** → Actual time taken (minutes)  
- **Partner Rating** → Customer rating of the delivery person  
    """)

cities = sorted(df["City"].unique().tolist())
selected_cities = st.sidebar.multiselect(
    "City", cities, default=cities[:7] if len(cities) > 7 else cities
)

weathers = sorted(df["Weather_conditions"].unique().tolist())
selected_weather = st.sidebar.multiselect("Weather", weathers, default=weathers)

traffics = sorted(df["Road_traffic_density"].unique().tolist())
selected_traffic = st.sidebar.multiselect("Traffic Density", traffics, default=traffics)

vehicles = sorted(df["Type_of_vehicle"].unique().tolist())
selected_vehicle = st.sidebar.multiselect("Vehicle Type", vehicles, default=vehicles)

festivals = sorted(df["Festival"].unique().tolist())
selected_festival = st.sidebar.multiselect("Festival", festivals, default=festivals)

min_date = df["Order_Date"].min().date()
max_date = df["Order_Date"].max().date()
date_range = st.sidebar.date_input("Date Range", [min_date, max_date])

# Apply filters
mask = (
    (df["City"].isin(selected_cities)) &
    (df["Weather_conditions"].isin(selected_weather)) &
    (df["Road_traffic_density"].isin(selected_traffic)) &
    (df["Type_of_vehicle"].isin(selected_vehicle)) &
    (df["Festival"].isin(selected_festival)) &
    (df["Order_Date"].dt.date >= date_range[0]) &
    (df["Order_Date"].dt.date <= date_range[1])
)
fdf = df[mask].copy()

# ---------------------------------------------------------
# HEADER
# ---------------------------------------------------------
st.markdown("""
    <div style="margin-bottom: 8px;">
        <h1 style="color:#0F172A; margin-bottom:6px; font-size:1.85rem; font-weight:700;">
            Zomato Last-Mile Delivery Intelligence
        </h1>
        <p style="color:#334155; font-size:1.05rem; margin-top:0;">
            Performance Monitoring &nbsp;•&nbsp; Operational Insights &nbsp;•&nbsp; Delivery Prediction
        </p>
    </div>
""", unsafe_allow_html=True)

# Download filtered data
csv_data = fdf.to_csv(index=False).encode("utf-8")
st.download_button(
    label="Download Filtered Data",
    data=csv_data,
    file_name=f"zomato_filtered_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
    mime="text/csv"
)

st.markdown("---")

# ---------------------------------------------------------
# KPI CARDS
# ---------------------------------------------------------
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Total Deliveries", f"{len(fdf):,}")
k2.metric("Avg Delivery Time", f"{fdf['Time_taken (min)'].mean():.1f} min")
k3.metric("Avg Distance", f"{fdf['distance_km'].mean():.2f} km")
k4.metric("Avg Partner Rating", f"{fdf['Delivery_person_Ratings'].mean():.2f}")
k5.metric("On-Time (≤ 30 min)", f"{(fdf['Time_taken (min)'] <= 30).mean() * 100:.1f}%")

st.markdown("")

# ---------------------------------------------------------
# CHART STYLING FUNCTION
# ---------------------------------------------------------
DARK_TEXT = "#0F172A"
PRIMARY = "#0D9488"
PALETTE = ["#0F172A", "#0D9488", "#14B8A6", "#F59E0B", "#F97316", "#EF4444"]

def style_chart(fig):
    fig.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(color=DARK_TEXT, size=13),
        title_font=dict(color=DARK_TEXT, size=16),
        legend_font=dict(color=DARK_TEXT),
        xaxis=dict(
            title_font=dict(color=DARK_TEXT),
            tickfont=dict(color=DARK_TEXT),
            gridcolor="#E2E8F0",
            linecolor="#CBD5E1"
        ),
        yaxis=dict(
            title_font=dict(color=DARK_TEXT),
            tickfont=dict(color=DARK_TEXT),
            gridcolor="#E2E8F0",
            linecolor="#CBD5E1"
        ),
        margin=dict(t=50, b=40, l=40, r=20)
    )
    return fig

# ---------------------------------------------------------
# TABS
# ---------------------------------------------------------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Overview",
    "Performance Drivers",
    "Partners",
    "Geography",
    "Prediction",
    "Documentation"
])

# ----- TAB 1: Overview -----
with tab1:
    c1, c2 = st.columns(2)
    with c1:
        daily = fdf.groupby(fdf["Order_Date"].dt.date).size().reset_index(name="Deliveries")
        fig = px.line(daily, x="Order_Date", y="Deliveries", title="Daily Delivery Volume")
        fig.update_traces(line_color=PRIMARY, line_width=2.5)
        st.plotly_chart(style_chart(fig), use_container_width=True)

    with c2:
        hourly = fdf.groupby("order_hour")["Time_taken (min)"].mean().reset_index()
        fig = px.bar(hourly, x="order_hour", y="Time_taken (min)", title="Average Delivery Time by Hour")
        fig.update_traces(marker_color=PRIMARY)
        st.plotly_chart(style_chart(fig), use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        speed_count = fdf["delivery_speed"].value_counts().reset_index()
        speed_count.columns = ["Category", "Count"]
        fig = px.pie(speed_count, values="Count", names="Category", hole=0.5,
                     title="Delivery Speed Distribution", color_discrete_sequence=PALETTE)
        st.plotly_chart(style_chart(fig), use_container_width=True)

    with c4:
        dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        dow = fdf.groupby("day_of_week")["Time_taken (min)"].mean().reindex(dow_order).reset_index()
        fig = px.bar(dow, x="day_of_week", y="Time_taken (min)", title="Average Time by Day of Week")
        fig.update_traces(marker_color=PRIMARY)
        st.plotly_chart(style_chart(fig), use_container_width=True)

# ----- TAB 2: Performance Drivers -----
with tab2:
    c1, c2 = st.columns(2)
    with c1:
        w = fdf.groupby("Weather_conditions")["Time_taken (min)"].mean().sort_values(ascending=False).reset_index()
        fig = px.bar(w, x="Weather_conditions", y="Time_taken (min)", title="Impact of Weather",
                     color="Time_taken (min)", color_continuous_scale=["#99F6E4", PRIMARY, "#134E4A"])
        st.plotly_chart(style_chart(fig), use_container_width=True)

    with c2:
        t = fdf.groupby("Road_traffic_density")["Time_taken (min)"].mean().sort_values(ascending=False).reset_index()
        fig = px.bar(t, x="Road_traffic_density", y="Time_taken (min)", title="Impact of Traffic",
                     color="Time_taken (min)", color_continuous_scale=["#99F6E4", PRIMARY, "#134E4A"])
        st.plotly_chart(style_chart(fig), use_container_width=True)

    pivot = fdf.pivot_table(values="Time_taken (min)", index="Weather_conditions",
                            columns="Road_traffic_density", aggfunc="mean")
    fig = px.imshow(pivot, text_auto=".1f", aspect="auto",
                    title="Weather × Traffic Interaction (Average Minutes)",
                    color_continuous_scale=["#CCFBF1", PRIMARY, "#134E4A"])
    st.plotly_chart(style_chart(fig), use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        v = fdf.groupby("Type_of_vehicle")["Time_taken (min)"].mean().reset_index()
        fig = px.bar(v, x="Type_of_vehicle", y="Time_taken (min)", title="Performance by Vehicle Type")
        fig.update_traces(marker_color=PRIMARY)
        st.plotly_chart(style_chart(fig), use_container_width=True)

    with c4:
        multi = fdf.groupby("multiple_deliveries")["Time_taken (min)"].mean().reset_index()
        fig = px.bar(multi, x="multiple_deliveries", y="Time_taken (min)", title="Impact of Multiple Deliveries")
        fig.update_traces(marker_color=PRIMARY)
        st.plotly_chart(style_chart(fig), use_container_width=True)

# ----- TAB 3: Partners -----
with tab3:
    c1, c2 = st.columns(2)
    with c1:
        fig = px.histogram(fdf, x="Delivery_person_Ratings", nbins=20, title="Partner Rating Distribution")
        fig.update_traces(marker_color=PRIMARY)
        st.plotly_chart(style_chart(fig), use_container_width=True)

    with c2:
        age_bins = pd.cut(fdf["Delivery_person_Age"], bins=[18, 25, 30, 35, 40, 55],
                          labels=["18-25", "26-30", "31-35", "36-40", "41+"])
        age_perf = fdf.groupby(age_bins, observed=True)["Time_taken (min)"].mean().reset_index()
        fig = px.bar(age_perf, x="Delivery_person_Age", y="Time_taken (min)", title="Average Time by Age Group")
        fig.update_traces(marker_color=PRIMARY)
        st.plotly_chart(style_chart(fig), use_container_width=True)

    fdf = fdf.copy()
    fdf["Rating_Band"] = pd.cut(fdf["Delivery_person_Ratings"],
                                 bins=[0, 3.5, 4.0, 4.5, 5.01],
                                 labels=["≤ 3.5", "3.6–4.0", "4.1–4.5", "4.6–5.0"])
    fig = px.box(fdf, x="Rating_Band", y="Time_taken (min)", title="Delivery Time by Partner Rating Band")
    fig.update_traces(marker_color=PRIMARY)
    st.plotly_chart(style_chart(fig), use_container_width=True)

# ----- TAB 4: Geography -----
with tab4:
    c1, c2 = st.columns(2)
    with c1:
        sample = fdf.sample(min(3500, len(fdf)))
        fig = px.scatter(sample, x="distance_km", y="Time_taken (min)",
                         color="Road_traffic_density", opacity=0.65,
                         title="Distance vs Delivery Time",
                         color_discrete_sequence=PALETTE)
        st.plotly_chart(style_chart(fig), use_container_width=True)

    with c2:
        city_perf = fdf.groupby("City").agg({
            "Time_taken (min)": "mean",
            "distance_km": "mean",
            "ID": "count"
        }).reset_index().rename(columns={"ID": "Deliveries"})
        city_perf = city_perf.sort_values("Time_taken (min)", ascending=False).head(12)
        fig = px.bar(city_perf, x="City", y="Time_taken (min)",
                     title="Cities with Highest Average Delivery Time",
                     hover_data=["distance_km", "Deliveries"])
        fig.update_traces(marker_color=PRIMARY)
        st.plotly_chart(style_chart(fig), use_container_width=True)

# ----- TAB 5: Prediction -----
with tab5:
    st.markdown("##### Model trained on the full cleaned dataset for reliable results")

    @st.cache_resource
    def train_model(full_data):
        model_df = full_data.copy()
        le_weather = LabelEncoder()
        le_traffic = LabelEncoder()
        le_vehicle = LabelEncoder()
        le_festival = LabelEncoder()

        model_df["weather_enc"] = le_weather.fit_transform(model_df["Weather_conditions"])
        model_df["traffic_enc"] = le_traffic.fit_transform(model_df["Road_traffic_density"])
        model_df["vehicle_enc"] = le_vehicle.fit_transform(model_df["Type_of_vehicle"])
        model_df["festival_enc"] = le_festival.fit_transform(model_df["Festival"])

        features = [
            "distance_km", "Delivery_person_Age", "Delivery_person_Ratings",
            "Vehicle_condition", "multiple_deliveries", "order_hour",
            "weather_enc", "traffic_enc", "vehicle_enc", "festival_enc"
        ]

        X = model_df[features]
        y = model_df["Time_taken (min)"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = RandomForestRegressor(n_estimators=120, max_depth=14, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)

        importance = pd.DataFrame({
            "Feature": features,
            "Importance": model.feature_importances_
        }).sort_values("Importance", ascending=False)

        return model, le_weather, le_traffic, le_vehicle, le_festival, mae, r2, importance

    with st.spinner("Loading prediction model..."):
        model, le_w, le_t, le_v, le_f, mae, r2, importance = train_model(df)

    m1, m2 = st.columns(2)
    m1.metric("Model Accuracy (R²)", f"{r2:.1%}")
    m2.metric("Average Error", f"{mae:.1f} min")

    fig = px.bar(importance, x="Importance", y="Feature", orientation="h",
                 title="Feature Importance", color="Importance",
                 color_continuous_scale=["#99F6E4", PRIMARY, "#134E4A"])
    st.plotly_chart(style_chart(fig), use_container_width=True)

    st.markdown("##### What-If Scenario")
    c1, c2, c3 = st.columns(3)
    with c1:
        dist = st.slider("Distance (km)", 0.5, 20.0, 3.0, 0.1)
        age = st.slider("Partner Age", 18, 50, 28)
        rating = st.slider("Partner Rating", 2.5, 5.0, 4.6, 0.1)
    with c2:
        vehicle_cond = st.slider("Vehicle Condition (0–2)", 0, 2, 2)
        multi = st.selectbox("Multiple Deliveries", [0, 1, 2, 3])
        hour = st.slider("Order Hour", 0, 23, 19)
    with c3:
        weather = st.selectbox("Weather", sorted(df["Weather_conditions"].unique()))
        traffic = st.selectbox("Traffic", sorted(df["Road_traffic_density"].unique()))
        vehicle = st.selectbox("Vehicle Type", sorted(df["Type_of_vehicle"].unique()))
        festival = st.selectbox("Festival", sorted(df["Festival"].unique()))

    try:
        input_data = np.array([[
            dist, age, rating, vehicle_cond, multi, hour,
            le_w.transform([weather])[0],
            le_t.transform([traffic])[0],
            le_v.transform([vehicle])[0],
            le_f.transform([festival])[0]
        ]])
        prediction = model.predict(input_data)[0]
        st.markdown(f"### Predicted Delivery Time: **{prediction:.1f} minutes**")

        if prediction <= 25:
            st.success("Fast delivery expected")
        elif prediction <= 35:
            st.info("Average delivery time")
        else:
            st.warning("Longer than usual — consider adding buffer time")
    except Exception:
        st.error("Unable to generate prediction with the current inputs")

# ----- TAB 6: Documentation -----
with tab6:
    st.markdown("### System Documentation")

    if DOCS_PATH.exists():
        with open(DOCS_PATH, "r", encoding="utf-8") as f:
            docs_content = f.read()
        st.markdown(docs_content)

        st.download_button(
            label="Download Documentation (Markdown)",
            data=docs_content,
            file_name="Zomato_System_Documentation.md",
            mime="text/markdown"
        )
        st.info("Tip: Open the downloaded file or use your browser’s Print → Save as PDF to create a professional PDF version.")
    else:
        st.warning("Documentation file not found. Please ensure `documentation/System_Documentation.md` exists.")

# ---------------------------------------------------------
# FOOTER + SUMMARY REPORT
# ---------------------------------------------------------
st.markdown("---")
st.markdown("##### Export Summary Report")

report = f"""# Zomato Last-Mile Delivery Intelligence Report
Generated: {datetime.now().strftime('%d %B %Y, %I:%M %p')}

## Performance Snapshot
- Total Deliveries: {len(fdf):,}
- Average Delivery Time: {fdf['Time_taken (min)'].mean():.1f} min
- Average Distance: {fdf['distance_km'].mean():.2f} km
- Average Partner Rating: {fdf['Delivery_person_Ratings'].mean():.2f}
- On-Time Rate (≤30 min): {(fdf['Time_taken (min)'] <= 30).mean()*100:.1f}%

## Key Insights
### Speed Distribution
{fdf['delivery_speed'].value_counts().to_string()}

### Slowest Cities
{fdf.groupby('City')['Time_taken (min)'].mean().sort_values(ascending=False).head(5).round(1).to_string()}

### Traffic Impact
{fdf.groupby('Road_traffic_density')['Time_taken (min)'].mean().sort_values(ascending=False).round(1).to_string()}

### Weather Impact
{fdf.groupby('Weather_conditions')['Time_taken (min)'].mean().sort_values(ascending=False).round(1).to_string()}

## Recommendations
1. Prioritize high-traffic + adverse weather scenarios for dynamic ETA and incentives.
2. Focus operational support on the slowest performing cities.
3. Review multi-delivery assignment rules during peak hours.
4. Leverage higher-rated partners for time-sensitive orders.
"""

st.download_button(
    label="Download Summary Report",
    data=report,
    file_name=f"Zomato_Delivery_Report_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
    mime="text/markdown"
)

st.caption("Zomato Last-Mile Delivery Intelligence  •  Built for operational decision support")