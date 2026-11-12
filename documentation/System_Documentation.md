# Zomato Last-Mile Delivery Intelligence

## System Documentation

---

### 1. Introduction

This document explains the Zomato Last-Mile Delivery Intelligence system in detail. It is written for both technical and non-technical readers — operations managers, analysts, and decision-makers who need to understand what the system does, how it works, and how it can support better operational decisions.

The system was designed to turn raw delivery data into clear, usable insights without requiring advanced technical knowledge from the end user.

---

### 2. Purpose of the System

The main goal of this dashboard is simple:

**Help teams understand delivery performance and make better decisions using data.**

Specifically, the system aims to:

- Show how the delivery operation is performing overall
- Identify the main reasons behind delays
- Highlight differences across cities, weather conditions, traffic levels, and vehicle types
- Provide a practical way to estimate delivery time under different scenarios
- Reduce the time spent on manual data analysis in spreadsheets

---

### 3. What the System Covers

**Included in the current version:**
- Performance monitoring through key metrics
- Interactive filtering across multiple dimensions
- Analysis of factors that influence delivery time
- Delivery partner performance insights
- City-level comparison
- Predictive delivery time estimation
- Ability to export filtered data and summary reports
- Built-in documentation

**Not included in the current version:**
- Real-time live tracking of ongoing deliveries
- Customer-facing features
- Financial or revenue analysis
- Direct connection to live production databases

---

### 4. How the System is Structured

The application is organized into clear sections:

1. **Filters (Sidebar)** – Users can narrow down the data by city, weather, traffic, vehicle type, festival, and date range.
2. **KPI Cards** – Quick snapshot of the most important numbers.
3. **Overview Tab** – Trends over time and general performance patterns.
4. **Performance Drivers Tab** – Shows which factors (weather, traffic, etc.) have the strongest impact on delivery time.
5. **Partners Tab** – Focuses on delivery partner ratings and performance.
6. **Geography Tab** – Looks at distance and city-level differences.
7. **Prediction Tab** – Allows users to test different scenarios and get an estimated delivery time.
8. **Documentation Tab** – Contains this full system documentation.

---

### 5. About the Data

The dashboard uses a cleaned version of historical Zomato delivery records.

- Original number of records: 45,584
- Records after cleaning: 36,474

During cleaning, incomplete rows, invalid coordinates, and unrealistic distances were removed. Missing values were handled carefully, and additional useful fields (such as calculated distance and order hour) were created.

The data includes information about delivery partners, order timing, weather, traffic conditions, vehicle type, and actual delivery time.

---

### 6. Predictive Model

The system includes a machine learning model that estimates how long a delivery is likely to take.

- **Model used:** Random Forest Regressor
- **What it predicts:** Delivery time in minutes
- **Main factors it considers:** Distance, partner age and rating, vehicle condition, number of multiple deliveries, time of day, weather, traffic, vehicle type, and whether it is a festival day

**Performance on the full cleaned dataset:**
- The model explains around 80% of the variation in delivery time (R² ≈ 80%)
- On average, the prediction is off by only about 3.1 minutes

The model is trained on the complete cleaned dataset so that predictions remain stable even when users apply filters on the dashboard.

---

### 7. How to Use the Dashboard

1. Start by adjusting the filters on the left side according to what you want to analyze.
2. Look at the KPI cards at the top for a quick overview.
3. Move through the different tabs to explore trends, causes of delays, partner performance, and geographic patterns.
4. Use the Prediction tab if you want to test a specific scenario (for example: long distance + heavy traffic + rainy weather).
5. Download the filtered data or a summary report whenever needed.
6. Refer to the Documentation tab if you need a fuller explanation of the system.

---

### 8. Business Value

This dashboard helps teams:

- Quickly see where performance is strong or weak
- Understand the real impact of weather and traffic
- Identify cities or conditions that need more attention
- Test ideas before making operational changes
- Spend less time preparing reports manually

In short, it supports faster and more confident decision-making.

---

### 9. Technical Notes

The system is built with:

- Python
- Streamlit (for the interactive interface)
- Pandas and NumPy (for data handling)
- Plotly (for visualizations)
- Scikit-learn (for the predictive model)

To run it locally:

```bash
pip install -r requirements.txt
streamlit run app.py

```

10. Maintenance and Future Improvements
Current maintenance needs:

Update the dataset file when newer cleaned data becomes available
Occasionally check that the predictive model still performs well

Possible future improvements:

Connecting to live data sources
Adding user login and access control
Creating automated scheduled reports
Expanding the prediction capabilities
Making the interface more mobile-friendly


11. Closing Note
This system was designed to be practical, understandable, and useful. It does not try to do everything — instead, it focuses on giving clear visibility into last-mile delivery performance and supporting better day-to-day and strategic decisions.