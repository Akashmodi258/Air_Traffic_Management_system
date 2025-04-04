import pandas as pd
import random
import uuid
from datetime import datetime, timedelta

city_airport_map = {
    "New York": "JFK International Airport",
    "Los Angeles": "LAX International Airport",
    "Chicago": "O'Hare International Airport",
    "London": "Heathrow Airport",
    "Tokyo": "Narita International Airport",
    "Paris": "Charles de Gaulle Airport",
    "Dubai": "Dubai International Airport",
    "Sydney": "Sydney Kingsford Smith Airport",
    "Mumbai": "Chhatrapati Shivaji Maharaj International Airport",
    "Singapore": "Changi Airport"
}

weather_conditions = [
    "Sunny", "Cloudy", "Rainy", "Snowy", "Windy", "Foggy", "Thunderstorm",
    "Hail", "Stormy", "Tornado", "Hurricane", "Freezing Rain", "Dust Storm"
]

data = []
start_date = datetime(2025, 1, 1)
end_date = datetime(2025, 12, 31)

def calculate_delay(weather):
    delays = {
        "Sunny": 0,
        "Cloudy": 10,
        "Rainy": 30,
        "Snowy": 60,
        "Windy": 45,
        "Foggy": 50,
        "Thunderstorm": 90,
        "Hail": 120,
        "Stormy": 180,
        "Tornado": 240,
        "Hurricane": 300,
        "Freezing Rain": 150,
        "Dust Storm": 100
    }
    return delays.get(weather, 0)

for _ in range(10000):
    flight_id = str(uuid.uuid4())[:8].upper()
    flight_type = random.choice(["Arrival", "Departure"])
    city = random.choice(list(city_airport_map.keys()))
    airport = city_airport_map[city]
    airport_capacity = random.randint(50, 500)

    random_days = random.randint(0, (end_date - start_date).days)
    flight_date = start_date + timedelta(days=random_days)
    flight_time = f"{random.randint(0, 23):02d}:{random.randint(0, 59):02d}"

    weather_condition = random.choice(weather_conditions)

    delay = calculate_delay(weather_condition)

    flight_time_obj = datetime.strptime(f"{flight_date.date()} {flight_time}", "%Y-%m-%d %H:%M")
    delayed_time_obj = flight_time_obj + timedelta(minutes=delay)
    delayed_flight_time = delayed_time_obj.strftime("%H:%M")

    data.append([flight_id, flight_type, city, airport, airport_capacity, flight_date.date(), flight_time, weather_condition, delayed_flight_time])

df = pd.DataFrame(data, columns=["Flight_ID", "Flight_Type", "City", "Airport", "Airport_Capacity", "Date", "Time", "Weather_Condition", "Delayed_Time"])

df.to_excel("air_traffic_dataset_with_weather.xlsx", index=False)