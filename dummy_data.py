import numpy as np
import pandas as pd
from datetime import datetime

# Generate dummy data
dummy_data = [
    {'weather_condition': 'Sunny', 'accident_occurred': False, 'Airport_Capacity': 100, 'flight_type': 'Departure', 'city': 'Bangalore', 'Airport': 'BLR', 'Scheduled_DateTime': datetime(2025, 4, 3, 12, 30), 'Delayed_Time': 0},
    {'weather_condition': 'Cloudy', 'accident_occurred': True, 'Airport_Capacity': 90, 'flight_type': 'Departure', 'city': 'Bangalore', 'Airport': 'HAL', 'Scheduled_DateTime': datetime(2025, 4, 3, 14, 45), 'Delayed_Time': 60},
    {'weather_condition': 'Rainy', 'accident_occurred': False, 'Airport_Capacity': 120, 'flight_type': 'Arrival', 'city': 'Bangalore', 'Airport': 'BLR', 'Scheduled_DateTime': datetime(2025, 4, 3, 9, 15), 'Delayed_Time': 30},
    {'weather_condition': 'Snowy', 'accident_occurred': True, 'Airport_Capacity': 80, 'flight_type': 'Arrival', 'city': 'Bangalore', 'Airport': 'HAL', 'Scheduled_DateTime': datetime(2025, 4, 3, 6, 0), 'Delayed_Time': 120},
    {'weather_condition': 'Windy', 'accident_occurred': False, 'Airport_Capacity': 110, 'flight_type': 'Departure', 'city': 'Bangalore', 'Airport': 'BLR', 'Scheduled_DateTime': datetime(2025, 4, 3, 16, 20), 'Delayed_Time': 15},
    {'weather_condition': 'Sunny', 'accident_occurred': False, 'Airport_Capacity': 105, 'flight_type': 'Arrival', 'city': 'Bangalore', 'Airport': 'HAL', 'Scheduled_DateTime': datetime(2025, 4, 3, 11, 0), 'Delayed_Time': 0},
    {'weather_condition': 'Cloudy', 'accident_occurred': False, 'Airport_Capacity': 95, 'flight_type': 'Departure', 'city': 'Bangalore', 'Airport': 'BLR', 'Scheduled_DateTime': datetime(2025, 4, 3, 13, 30), 'Delayed_Time': 20},
    {'weather_condition': 'Rainy', 'accident_occurred': True, 'Airport_Capacity': 115, 'flight_type': 'Arrival', 'city': 'Bangalore', 'Airport': 'HAL', 'Scheduled_DateTime': datetime(2025, 4, 3, 8, 45), 'Delayed_Time': 90},
    {'weather_condition': 'Snowy', 'accident_occurred': False, 'Airport_Capacity': 85, 'flight_type': 'Departure', 'city': 'Bangalore', 'Airport': 'BLR', 'Scheduled_DateTime': datetime(2025, 4, 3, 5, 30), 'Delayed_Time': 45},
    {'weather_condition': 'Windy', 'accident_occurred': True, 'Airport_Capacity': 108, 'flight_type': 'Arrival', 'city': 'Bangalore', 'Airport': 'HAL', 'Scheduled_DateTime': datetime(2025, 4, 3, 15, 10), 'Delayed_Time': 75},
]

# Convert to DataFrame
df_dummy = pd.DataFrame(dummy_data)

# Save to CSV
df_dummy.to_csv('dummy_flights.csv', index=False)

print("Dummy flight data generated and saved to 'dummy_flights.csv'.")