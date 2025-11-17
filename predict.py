import pickle

from fastapi import FastAPI
import uvicorn
import math
import h3
from pydantic import BaseModel, validator
from datetime import datetime
import pandas as pd
# NYC bounding box (approximate)
NY_LAT_MIN = 40.4774
NY_LAT_MAX = 40.9176
NY_LON_MIN = -74.2591
NY_LON_MAX = -73.7004

# --------------------------
# Utils: distance + h3
# --------------------------

def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # km
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(math.radians(lat1)) *
         math.cos(math.radians(lat2)) *
         math.sin(dlon / 2) ** 2)
    return R * 2 * math.asin(math.sqrt(a))


def compute_h3(lat, lon, resolution=8):
    return int(h3.latlng_to_cell(lat, lon, resolution), 16) 
    

class Trip(BaseModel):
    timestamp: datetime 
    pickup_latitude: float
    pickup_longitude: float
    dropoff_latitude: float
    dropoff_longitude: float
    passenger_count: int

    @validator("timestamp")
    def timestamp_hour_valid(cls, v):
        hour = v.hour
        if not (0 <= hour <= 23):
            raise ValueError("datetime hour must be between 0 and 23")
        return v

    @validator("passenger_count")
    def passenger_count_limit(cls, v):
        if v < 1:
            raise ValueError("passenger_count must be >= 1")
        if v > 5:
            raise ValueError("passenger_count cannot be greater than 5")
        return v

    @validator("pickup_latitude", "dropoff_latitude")
    def latitude_in_nyc(cls, v):
        if not (NY_LAT_MIN <= v <= NY_LAT_MAX):
            raise ValueError(f"latitude must be within NYC bounds [{NY_LAT_MIN}, {NY_LAT_MAX}]")
        return v

    @validator("pickup_longitude", "dropoff_longitude")
    def longitude_in_nyc(cls, v):
        if not (NY_LON_MIN <= v <= NY_LON_MAX):
            raise ValueError(f"longitude must be within NYC bounds [{NY_LON_MIN}, {NY_LON_MAX}]")
        return v

    class Config:
        anystr_strip_whitespace = True
        schema_extra = {
            "example": {
                "timestamp": "2025-01-20T15:42:00",
                "pickup_latitude": 40.7580,
                "pickup_longitude": -73.9855,
                "dropoff_latitude": 40.6413,
                "dropoff_longitude": -73.7781,
                "passenger_count": 2
            }
        }


class PredictResponse(BaseModel):
    fare_amount: float


def preprocess(trip_dict):

    # Extract hour
    ts = trip_dict["timestamp"]
    hour = ts.hour
    month = ts.month
    day_of_week = ts.weekday()

    # Distance
    dist = haversine(
        trip_dict["pickup_latitude"],
        trip_dict["pickup_longitude"],
        trip_dict["dropoff_latitude"],
        trip_dict["dropoff_longitude"],
    )

    # H3
    h3_pickup = compute_h3(
        trip_dict["pickup_latitude"],
        trip_dict["pickup_longitude"],
        resolution=8,
    )

    h3_dropoff = compute_h3(
        trip_dict["dropoff_latitude"],
        trip_dict["dropoff_longitude"],
        resolution=8,
    )

    # Build final feature vector
    df = pd.DataFrame([{
        "distance_km": dist,
        "hour": hour,
        "day_of_week": day_of_week,
        "month": month,
        "passenger_count": trip_dict["passenger_count"],
        "is_rush_hour":int(hour in [7, 8, 9, 16, 17, 18]),
        "pickup_h3_int": h3_pickup,
        "dropoff_h3_int": h3_dropoff,
    }])

    return df

app = FastAPI(title="trip-fare-prediction")

with open('model.bin', 'rb') as f_in:
    pipeline = pickle.load(f_in)

print("Loaded pipeline:", type(pipeline))


def predict_single(trip_dict):
    X = preprocess(trip_dict)
    pred = pipeline.predict(X)[0]  
    return float(pred)



@app.post("/predict")
def predict(trip: Trip) -> PredictResponse:
    fare = predict_single(trip.model_dump())

    return PredictResponse(
        fare_amount=fare,
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9696)



