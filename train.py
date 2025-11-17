import pandas as pd
import numpy as np
import pickle

from xgboost import XGBRegressor

from h3 import latlng_to_cell


def haversine(lon1, lat1, lon2, lat2):
    R = 6371  # km
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

def load_data():
    df = pd.read_csv("uber.csv.gz", compression='gzip')
    # Select columns
    df = df[['fare_amount', 'pickup_datetime', 'pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude', 'passenger_count']]
    # remove unrealistic fares
    df = df[(df["fare_amount"] > 0) & (df["fare_amount"] < 200)]

    # latitude/longitude ranges (NYC approx)
    df = df[
        (df["pickup_latitude"].between(40, 42)) &
        (df["pickup_longitude"].between(-75, -72)) &
        (df["dropoff_latitude"].between(40, 42)) &
        (df["dropoff_longitude"].between(-75, -72))
    ]
    # Parse datetime
    df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])
    df["hour"] = df["pickup_datetime"].dt.hour
    df["day_of_week"] = df["pickup_datetime"].dt.dayofweek
    df["month"] = df["pickup_datetime"].dt.month
    df["is_weekend"] = df["day_of_week"].isin([5,6]).astype(int)
    df["is_rush_hour"] = df["hour"].isin([7,8,9,16,17,18]).astype(int)

    # Distance (Haversine)
    df["distance_km"] = haversine(
        df["pickup_longitude"], df["pickup_latitude"],
        df["dropoff_longitude"], df["dropoff_latitude"]
    )
    df = df[df["distance_km"] < 50]  # remove wild values

    # H3 location encoding
    resolution = 8
    df["pickup_h3"] = df.apply(
        lambda r: latlng_to_cell(r["pickup_latitude"], r["pickup_longitude"], resolution), axis=1
    )
    df["dropoff_h3"] = df.apply(
        lambda r: latlng_to_cell(r["dropoff_latitude"], r["dropoff_longitude"], resolution), axis=1
    )
    df["pickup_h3_int"] = df["pickup_h3"].apply(lambda x: int(x, 16))
    df["dropoff_h3_int"] = df["dropoff_h3"].apply(lambda x: int(x, 16))
    df = df.drop(columns=["pickup_h3", "dropoff_h3"])

    df["log_fare"] = np.log1p(df["fare_amount"])
    df = df.drop(columns=["fare_amount"])
    
    return df

# In[25]:
def train_model(df):
    features = [
        "distance_km",
        "hour",
        "day_of_week",
        "month",
        "passenger_count",
        "is_rush_hour",
        "pickup_h3_int",
        "dropoff_h3_int"
    ]

    X = df[features]

    y = df["log_fare"]

    model = XGBRegressor(
        n_estimators=400,
        learning_rate=0.03,
        max_depth=4,
        min_child_weight=1,
        subsample=1.0,
        colsample_bytree=0.8,
        random_state=42,
        tree_method="hist",
    )

    model.fit(X, y)
    return model

def save_model(pipeline, output_file):
    with open(output_file, 'wb') as f_out:
        pickle.dump(pipeline, f_out)


df = load_data()
pipeline = train_model(df)
save_model(pipeline, 'model.bin')

print('Model saved to model.bin')