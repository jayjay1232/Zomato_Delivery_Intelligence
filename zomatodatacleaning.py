import pandas as pd
import numpy as np
from math import radians, cos, sin, asin, sqrt
import warnings
warnings.filterwarnings('ignore')

# ====================== 1. LOAD DATA ======================
df = pd.read_csv("zomato_data.csv")   # ← FIXED: use read_csv for .csv files

print("Original shape:", df.shape)
print("\nMissing values before cleaning:")
print(df.isnull().sum())
print("\nColumn names:", df.columns.tolist())

# ====================== 2. BASIC CLEANING ======================
df.columns = df.columns.str.strip()

# Drop completely empty rows
df = df.dropna(how='all')

# Remove exact duplicate rows
df = df.drop_duplicates()

# ====================== 3. HANDLE MISSING VALUES ======================
critical_cols = [
    'Time_taken (min)', 
    'Restaurant_latitude', 'Restaurant_longitude',
    'Delivery_location_latitude', 'Delivery_location_longitude'
]
df = df.dropna(subset=critical_cols)

# Numeric columns → median
df['Delivery_person_Age'] = pd.to_numeric(df['Delivery_person_Age'], errors='coerce')
df['Delivery_person_Ratings'] = pd.to_numeric(df['Delivery_person_Ratings'], errors='coerce')

df['Delivery_person_Age'] = df['Delivery_person_Age'].fillna(df['Delivery_person_Age'].median())
df['Delivery_person_Ratings'] = df['Delivery_person_Ratings'].fillna(df['Delivery_person_Ratings'].median())

# Categorical columns → mode or 'Unknown'
cat_cols = ['Weather_conditions', 'Road_traffic_density', 'City', 
            'Festival', 'Type_of_order', 'Type_of_vehicle']
for col in cat_cols:
    if col in df.columns:
        mode_val = df[col].mode()[0] if not df[col].mode().empty else 'Unknown'
        df[col] = df[col].fillna(mode_val)

# multiple_deliveries
df['multiple_deliveries'] = pd.to_numeric(df['multiple_deliveries'], errors='coerce').fillna(0)

# ====================== 4. DATA TYPE CONVERSION ======================
df['Delivery_person_Age'] = df['Delivery_person_Age'].astype(int)
df['Delivery_person_Ratings'] = df['Delivery_person_Ratings'].astype(float)
df['Time_taken (min)'] = pd.to_numeric(df['Time_taken (min)'], errors='coerce').astype(int)
df['Vehicle_condition'] = pd.to_numeric(df['Vehicle_condition'], errors='coerce').fillna(0).astype(int)
df['multiple_deliveries'] = df['multiple_deliveries'].astype(int)

# Date
df['Order_Date'] = pd.to_datetime(df['Order_Date'], dayfirst=True, errors='coerce')

# ====================== 5. REMOVE INVALID GPS COORDINATES ======================
df = df[
    (df['Restaurant_latitude'] != 0) & 
    (df['Restaurant_longitude'] != 0) &
    (df['Delivery_location_latitude'] != 0) & 
    (df['Delivery_location_longitude'] != 0)
]

# Keep only reasonable Indian lat/long ranges
df = df[
    (df['Restaurant_latitude'].between(6, 37)) &
    (df['Restaurant_longitude'].between(68, 98))
]

# ====================== 6. FEATURE ENGINEERING ======================
def haversine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    return c * 6371  # km

df['distance_km'] = df.apply(
    lambda row: haversine(
        row['Restaurant_longitude'], row['Restaurant_latitude'],
        row['Delivery_location_longitude'], row['Delivery_location_latitude']
    ), axis=1
)

# Remove unrealistic distances
df = df[df['distance_km'] <= 50]

# Time features
df['order_hour'] = pd.to_datetime(df['Time_Orderd'], format='%H:%M', errors='coerce').dt.hour
df['day_of_week'] = df['Order_Date'].dt.day_name()
df['month'] = df['Order_Date'].dt.month
df['is_weekend'] = df['Order_Date'].dt.dayofweek >= 5

# Delivery speed category
df['delivery_speed'] = pd.cut(
    df['Time_taken (min)'],
    bins=[0, 20, 35, 100],
    labels=['Fast', 'Average', 'Slow']
)

# ====================== 7. FINAL CLEANING ======================
df = df.dropna(subset=['distance_km', 'order_hour'])
df = df.reset_index(drop=True)

print("\n===== CLEANING COMPLETE =====")
print("Final shape:", df.shape)
print("\nMissing values after cleaning:")
print(df.isnull().sum())
print("\nSample:")
print(df.head())

# Save
df.to_csv("zomato_cleaned.csv", index=False)
print("\nCleaned data saved as → zomato_cleaned.csv")