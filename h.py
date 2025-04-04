import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def create_dummy_flight_data(num_rows=5):
    """
    Creates dummy flight data that resembles real-world flight information.
    """
    now = datetime.now()
    past_date = now - timedelta(days=30)
    future_date = now + timedelta(days=30)

    flight_types = ['Domestic', 'International']
    cities = ['Bangalore', 'Delhi', 'Mumbai', 'Chennai', 'Hyderabad']
    airports = ['BLR', 'DEL', 'BOM', 'MAA', 'HYD', 'GOI', 'PNQ'] # Added more airports
    weather_conditions = ['Sunny', 'Cloudy', 'Rainy', 'Stormy', 'Foggy']
    airport_capacities = np.random.randint(50, 300, size=len(airports))
    airport_capacity_dict = dict(zip(airports, airport_capacities))

    dummy_data = []
    for i in range(num_rows):
        flight_id = f"DF{np.random.randint(100, 999)}"
        flight_type = np.random.choice(flight_types)
        city = np.random.choice(cities)
        airport = np.random.choice(airports)
        weather = np.random.choice(weather_conditions)
        capacity = airport_capacity_dict.get(airport, np.random.randint(100, 200)) # Get capacity or random if not in dict

        # Generate a random datetime within a reasonable range
        random_time = past_date + (future_date - past_date) * np.random.rand()
        scheduled_datetime = random_time.strftime('%Y-%m-%d %H:%M:%S')

        # Introduce some realistic variation in original delay (some flights might not have a delay)
        delayed_time = np.random.randint(0, 60) if np.random.rand() < 0.7 else 0

        dummy_data.append({
            'Flight_ID': flight_id,
            'Flight_Type': flight_type,
            'City': city,
            'Airport': airport,
            'Weather_Condition': weather,
            'Airport_Capacity': capacity,
            'Date': scheduled_datetime.split()[0],
            'Time': scheduled_datetime.split()[1],
            'Scheduled_DateTime': scheduled_datetime,
            'Delayed_Time': delayed_time
        })

    return pd.DataFrame(dummy_data)

# --- Configuration for the dummy data ---
NUM_ROWS = 10  # You can change the number of rows of dummy data you want
OUTPUT_CSV_FILE = 'dummy_flights.csv' # Name of the CSV file to save

# --- Create the dummy data ---
dummy_df = create_dummy_flight_data(num_rows=NUM_ROWS)

# --- Save the dummy DataFrame to a CSV file ---
dummy_df.to_csv(OUTPUT_CSV_FILE, index=False)

# --- Optional: Print confirmation message and display the first few rows ---
print(f"Dummy flight data with {NUM_ROWS} rows saved to '{OUTPUT_CSV_FILE}'")
print("\nFirst 5 rows of the generated dummy data:")
print(dummy_df.head())