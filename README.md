# Zomato Last-Mile Delivery Intelligence

**Interactive Analytics & Predictive Dashboard for Last-Mile Delivery Operations**

---

### Overview

This project is an interactive analytics dashboard built to help understand and improve last-mile delivery performance. It turns historical delivery data into clear insights that operations teams and managers can actually use.

The dashboard allows users to explore delivery performance, identify what causes delays, evaluate delivery partners, and even estimate how long a delivery might take under different conditions.

---

### Key Features

- Easy-to-use filters (City, Weather, Traffic, Vehicle Type, Festival, Date Range)
- Clear KPI cards for quick performance overview
- Trend analysis (daily volume, hourly patterns, day of the week)
- Deep dive into factors that affect delivery time (weather, traffic, vehicle type, etc.)
- Delivery partner performance insights
- City-level performance comparison
- Machine learning model that predicts delivery time
- What-If scenario tool
- Ability to download filtered data and summary reports
- Built-in system documentation

---

### Project Structure
Zomato_Delivery_Intelligence/

├── app.py
├── zomatodatacleaning.py
├── requirements.txt
├── README.md
├── data/
│   └── zomato_cleaned.csv
│    └── zomato_data.csv
└── documentation/
    └── System_Documentation.md


---

### Dataset

- **File used:** `data/zomato_cleaned.csv`
- Original records: 45,584
- Cleaned records used in the dashboard: 36,474

---

### Predictive Model

- Algorithm: Random Forest Regressor
- Target: Delivery Time (in minutes)
- Performance on full cleaned data:
  - R² Score ≈ 80%
  - Average Error ≈ 3.1 minutes

---

### How to Run Locally

1. Clone the repository
2. Install dependencies:

```bash
pip install -r requirements.txt}
3. Place your cleaned dataset in the data/ folder as zomato_cleaned.csv
4. Run the application: 
   streamlit run app.py
```

DocumentationFull system documentation is available in the documentation folder and can also be viewed directly inside the dashboard under the Documentation tab.

Technology Stack

Python
Streamlit
Pandas & NumPy
Plotly
Scikit-learn

Purpose
This dashboard was built to demonstrate how data can be turned into practical insights for last-mile delivery operations — helping teams move from gut feeling to data-supported decisions.




