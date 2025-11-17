import requests

url = 'http://localhost:9696/predict'

trip = {
    "timestamp": "2025-01-20T15:42:00",
    "pickup_latitude": 40.7580,
    "pickup_longitude": -73.9855,
    "dropoff_latitude": 40.6413,
    "dropoff_longitude": -73.7781,
    "passenger_count": 2
}

response = requests.post(url, json=trip)
print(response)
predictions = response.json()

print(predictions)