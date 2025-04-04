# app.py (Corrected Full Script)
import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf # Make sure tensorflow is imported
import joblib
from datetime import timedelta
import os # Import os to check file existence

# --- Page Configuration (MUST BE THE FIRST STREAMLIT COMMAND) ---
# Moved this line to the top right after imports
st.set_page_config(layout="wide")

# --- Configuration ---
MODEL_PATH = 'lstm_rescheduler_model.h5' # Or folder path if SavedModel format
PREPROCESSED_CSV_FILE = 'preprocessed_flights.csv' # Needed for structure and inverse transforms maybe
ORIGINAL_DATA_FILE = 'air_traffic_dataset_with_weather.xlsx' # Load original data for display and capacity checks
DUMMY_DATA_FILE = 'dummy_flights.csv' # Path to the dummy data CSV
ENCODER_FILE = 'onehot_encoder.joblib'
SCALER_FILE = 'standard_scaler.joblib'
TARGET_SCALER_FILE = 'target_scaler.joblib'

# Define expected features for the model (must match training)
# These should ideally be derived from the saved encoder/scaler if possible,
# but defining them manually based on the preprocessing script is okay if consistent.
CAT_FEATURES = ['Flight_Type', 'City', 'Airport', 'Weather_Condition']
NUM_FEATURES = ['Airport_Capacity', 'Hour', 'Minute', 'DayOfWeek', 'Month', 'DayOfYear']

# --- Define Bad Weather Conditions for Diversion ---
BAD_WEATHER = ['Rainy', 'Stormy', 'Foggy', 'Freezing Rain', 'Dust Storm', 'Snowy']

# --- Load Artifacts ---
@st.cache_resource # Cache model and preprocessors
def load_artifacts():
    # Check if files exist before trying to load
    required_files = [MODEL_PATH, ENCODER_FILE, SCALER_FILE, TARGET_SCALER_FILE]
    missing_files = [f for f in required_files if not os.path.exists(f)]
    if missing_files:
        st.error(f"Error: Missing required artifact files: {', '.join(missing_files)}")
        st.error("Please ensure preprocessing and training scripts ran successfully.")
        return None, None, None, None, None

    try:
        # *** FIX: Added custom_objects to handle standard mse/mae ***
        model = tf.keras.models.load_model(
            MODEL_PATH,
            custom_objects={
                'mse': tf.keras.losses.MeanSquaredError(),
                'mae': tf.keras.metrics.MeanAbsoluteError()
            }
        )
        encoder = joblib.load(ENCODER_FILE)
        scaler = joblib.load(SCALER_FILE)
        target_scaler = joblib.load(TARGET_SCALER_FILE)

        # Derive feature names expected by the model from the encoder/scaler
        try:
            encoded_feature_names = encoder.get_feature_names_out(CAT_FEATURES)
            # Combine scaled numerical feature names + encoded categorical feature names
            all_feature_names = NUM_FEATURES + encoded_feature_names.tolist()
            print("DEBUG: Successfully derived feature names:", all_feature_names) # Debug print
        except Exception as e:
            st.error(f"Error deriving feature names from encoder: {e}")
            st.error("Ensure categorical features match those used in preprocessing.")
            return None, None, None, None, None

        return model, encoder, scaler, target_scaler, all_feature_names
    except Exception as e:
        st.error(f"Error loading necessary artifacts: {e}")
        st.error("Ensure the model file format is correct and compatible.")
        # Provide more detail if possible, e.g., print traceback for debugging
        import traceback
        st.error("Traceback:")
        st.code(traceback.format_exc())
        return None, None, None, None, None

@st.cache_data # Cache data loading
def load_data(use_dummy_data=False):
    if use_dummy_data:
        if not os.path.exists(DUMMY_DATA_FILE):
            st.error(f"Error: Dummy data file '{DUMMY_DATA_FILE}' not found. Please run the dummy data generation script.")
            return None, None
        try:
            df = pd.read_csv(DUMMY_DATA_FILE)
            # Ensure 'Scheduled_DateTime' is in datetime format
            if 'Scheduled_DateTime' in df.columns:
                df['Scheduled_DateTime'] = pd.to_datetime(df['Scheduled_DateTime'], errors='coerce')
            else:
                st.error("Error: 'Scheduled_DateTime' column not found in dummy data.")
                return None, None
            df['Delayed_Time'].fillna(0, inplace=True)
            airport_capacities = df.groupby('Airport')['Airport_Capacity'].first().to_dict() if 'Airport' in df.columns and 'Airport_Capacity' in df.columns else {}
            return df, airport_capacities
        except Exception as e:
            st.error(f"Error loading dummy flight data: {e}")
            import traceback
            st.error("Traceback:")
            st.code(traceback.format_exc())
            return None, None
    else:
        # Check if original data file exists
        if not os.path.exists(ORIGINAL_DATA_FILE):
            st.error(f"Error: Original data file '{ORIGINAL_DATA_FILE}' not found.")
            return None, None
        try:
            df_original = pd.read_excel(ORIGINAL_DATA_FILE)
            # Basic preprocessing needed for display/filtering
            try:
                datetime_str = df_original['Date'].astype(str) + ' ' + df_original['Time'].astype(str)
                df_original['Scheduled_DateTime'] = pd.to_datetime(datetime_str, errors='coerce')
            except Exception as dt_error:
                st.error(f"Error parsing Date/Time columns in original data: {dt_error}")
                df_original['Scheduled_DateTime'] = pd.NaT # Set to NaT if parsing fails

            df_original.dropna(subset=['Scheduled_DateTime'], inplace=True) # Drop rows if date/time essential
            df_original['Delayed_Time'].fillna(0, inplace=True) # Assuming 0 if NaN

            # Needed for capacity check / alternative airport suggestions
            # Create this carefully, handle potential duplicates or missing values
            if 'Airport' in df_original.columns and 'Airport_Capacity' in df_original.columns:
                # Drop rows with missing Airport or Capacity before creating the dictionary
                df_capacity = df_original.dropna(subset=['Airport', 'Airport_Capacity'])
                # Handle potential duplicate airports (e.g., take the first or average capacity)
                airport_capacities = df_capacity.drop_duplicates(subset=['Airport']).set_index('Airport')['Airport_Capacity'].to_dict()
            else:
                st.warning("Airport or Airport_Capacity column missing in original data. Cannot perform capacity checks.")
                airport_capacities = {}

            return df_original, airport_capacities
        except Exception as e:
            st.error(f"Error loading original flight data: {e}")
            import traceback
            st.error("Traceback:")
            st.code(traceback.format_exc())
            return None, None

# --- Load Resources ---
# These calls might trigger Streamlit elements due to caching, placed after set_page_config
use_dummy = st.sidebar.checkbox("Use Dummy Data for Prediction") # Checkbox to control data source
model, encoder, scaler, target_scaler, model_feature_names = load_artifacts()
df_original, airport_capacities = load_data(use_dummy_data=use_dummy) # Load data based on checkbox

# --- Initialize Session State ---
# Placed after loading functions, but before main UI elements
if 'recommendations' not in st.session_state:
    st.session_state.recommendations = [] # List to store past recommendations

# --- Streamlit App Layout ---
st.title("✈️ AI Flight Rescheduling Assistant")
st.markdown("""
Enter the current conditions for a specific airport to see potential rescheduling recommendations based on an LSTM prediction model.
""")

# --- Main Application Logic ---
# Check if artifacts and data loaded successfully before proceeding
if model and encoder and scaler and target_scaler and model_feature_names and (df_original is not None) and (airport_capacities is not None):

    # --- User Inputs ---
    col1, col2, col3 = st.columns(3)

    with col1:
        # Get unique airports from the loaded data source
        if df_original is not None and 'Airport' in df_original.columns:
            available_airports = sorted(df_original['Airport'].dropna().unique())
        else:
            st.warning("No airport data available.")
            available_airports = []
        if available_airports:
            selected_airport = st.selectbox("Select Airport:", available_airports, key="airport_select")
        else:
            st.warning("No airports available to select. Check the input data file or dummy data file.")
            selected_airport = None # Disable button later if no airport

    with col2:
        # Define plausible weather conditions (or get from data/encoder categories if possible)
        try:
            # Attempt to get weather conditions from the encoder if fitted
            weather_options = [cat.split('Weather_Condition_')[-1] for cat in encoder.get_feature_names_out(['Weather_Condition'])]
            if not weather_options: # Fallback if parsing fails or encoder issue
                raise ValueError("Could not parse weather options from encoder")
        except (AttributeError, ValueError, IndexError):
            # Fallback list if encoder isn't loaded properly or doesn't have weather
            st.warning("Could not dynamically load weather conditions. Using default list.")
            weather_options = ["Sunny", "Cloudy", "Rainy", "Stormy", "Foggy", "Freezing Rain", "Dust Storm", "Snowy"] # Add more as seen in data
        current_weather = st.selectbox("Select Current Weather:", sorted(list(set(weather_options))), key="weather_select")


    with col3:
        accident_occurred = st.radio("Runway Accident Occurred?", ("No", "Yes"), key="accident_radio")
        accident_status = 1 if accident_occurred == "Yes" else 0

    submit_button = st.button("Generate Recommendations", disabled=(selected_airport is None)) # Disable if no airport selected

    # --- Processing and Prediction ---
    if submit_button and selected_airport:
        st.info(f"Generating recommendations for **{selected_airport}** with Weather: **{current_weather}** and Accident: **{accident_occurred}**...")

        # Use the loaded df_original (which could be original or dummy data)
        relevant_flights = df_original[
            (df_original['Airport'] == selected_airport) &
            (pd.notna(df_original['Scheduled_DateTime']))
        ].copy()

        if relevant_flights.empty:
            st.warning(f"No flights with valid schedule data found for {selected_airport} in the selected dataset.")
        else:
            # Prepare data for prediction (apply preprocessing steps)
            prediction_input_list = []
            original_indices = relevant_flights.index # Keep track of original rows

            valid_flight_count = 0
            for index, flight in relevant_flights.iterrows():
                try:
                    # 1. Create DataFrame for the single flight
                    flight_data = pd.DataFrame([flight])

                    # 2. Extract Time Features (ensure Scheduled_DateTime is valid)
                    if pd.isna(flight_data['Scheduled_DateTime'].iloc[0]):
                        st.warning(f"Skipping Flight {flight.get('Flight_ID', index)} due to invalid Scheduled DateTime.")
                        continue # Skip this flight

                    flight_data['Hour'] = flight_data['Scheduled_DateTime'].dt.hour
                    flight_data['Minute'] = flight_data['Scheduled_DateTime'].dt.minute
                    flight_data['DayOfWeek'] = flight_data['Scheduled_DateTime'].dt.dayofweek
                    flight_data['Month'] = flight_data['Scheduled_DateTime'].dt.month
                    flight_data['DayOfYear'] = flight_data['Scheduled_DateTime'].dt.dayofyear

                    # 3. Override Weather Condition with user input
                    flight_data['Weather_Condition'] = current_weather
                    # Add accident status feature IF it was used in training (Needs careful handling)
                    # flight_data['Accident_At_Airport'] = accident_status # Example if trained with this

                    # 4. Select & Scale Numerical Features
                    # Ensure all required numerical features are present and not NaN
                    num_features_data = flight_data[NUM_FEATURES]
                    if num_features_data.isnull().values.any():
                        st.warning(f"Skipping Flight {flight.get('Flight_ID', index)} due to missing numerical features.")
                        # Optionally print which features are missing:
                        # print(f"Missing features in flight {index}: {num_features_data.columns[num_features_data.isnull().any()].tolist()}")
                        continue # Skip this flight

                    scaled_num_features = scaler.transform(num_features_data)
                    scaled_num_df = pd.DataFrame(scaled_num_features, columns=NUM_FEATURES)

                    # 5. Select & Encode Categorical Features
                    cat_features_data = flight_data[CAT_FEATURES]
                    if cat_features_data.isnull().values.any():
                        st.warning(f"Skipping Flight {flight.get('Flight_ID', index)} due to missing categorical features.")
                        continue # Skip this flight

                    encoded_cat_features = encoder.transform(cat_features_data)
                    encoded_cat_df = pd.DataFrame(encoded_cat_features, columns=encoder.get_feature_names_out(CAT_FEATURES))

                    # 6. Combine into a single feature vector
                    combined_features = pd.concat([scaled_num_df, encoded_cat_df], axis=1)

                    # 7. Reindex to ensure columns match model's expectation exactly
                    # Fill missing columns (e.g., if a category wasn't in this flight but was in training) with 0
                    # This relies on model_feature_names being loaded correctly
                    final_features_df = combined_features.reindex(columns=model_feature_names, fill_value=0)

                    # Check if reindexing created NaNs (shouldn't if fill_value=0 used properly)
                    if final_features_df.isnull().values.any():
                        st.error(f"NaNs found in feature vector for flight {flight.get('Flight_ID', index)} AFTER reindexing. Check feature names and logic.")
                        print("DEBUG: combined_features columns:", combined_features.columns)
                        print("DEBUG: model_feature_names:", model_feature_names)
                        print("DEBUG: final_features_df with NaNs:", final_features_df[final_features_df.isnull().any(axis=1)])
                        continue # Skip problematic row

                    prediction_input_list.append(final_features_df.values.flatten()) # Flatten since it's one row
                    valid_flight_count += 1

                except Exception as processing_error:
                    st.warning(f"Could not process flight {flight.get('Flight_ID', index)}: {processing_error}")
                    # Optionally print full traceback for debugging
                    # import traceback
                    # print(traceback.format_exc())
                    continue # Skip to next flight

            # --- Make Predictions (only if there are valid flights) ---
            if valid_flight_count > 0:
                # Convert list to numpy array for prediction
                prediction_input_array = np.array(prediction_input_list)
                # Reshape for LSTM: (samples, 1, features)
                prediction_input_lstm = prediction_input_array.reshape((prediction_input_array.shape[0], 1, prediction_input_array.shape[1]))

                try:
                    scaled_predictions = model.predict(prediction_input_lstm)
                    # Inverse transform predictions to get delay in minutes
                    predicted_delays = target_scaler.inverse_transform(scaled_predictions)
                    # Clip negative predictions to 0 and round
                    predicted_delays = np.maximum(predicted_delays, 0).flatten().round().astype(int)

                    # --- Generate Recommendations ---
                    # Filter original 'relevant_flights' to only include those successfully processed
                    processed_indices = relevant_flights.iloc[[i for i, flight_data in enumerate(prediction_input_list) if flight_data is not None]].index
                    results = relevant_flights.loc[processed_indices].copy() # Use .loc with original indices

                    # Add predictions (ensure length matches)
                    if len(predicted_delays) == len(results):
                        results['Predicted_Delay_Minutes'] = predicted_delays
                    else:
                        st.error(f"Prediction length mismatch: Expected {len(results)}, Got {len(predicted_delays)}. Cannot add predictions.")
                        # Handle error appropriately, maybe skip adding predictions or stop
                        predicted_delays = np.full(len(results), -1) # Placeholder for error
                        results['Predicted_Delay_Minutes'] = predicted_delays


                    # Calculate new estimated time (handle potential errors)
                    results['Adjusted_DateTime'] = results.apply(
                        lambda row: row['Scheduled_DateTime'] + timedelta(minutes=row['Predicted_Delay_Minutes']) if pd.notna(row['Scheduled_DateTime']) and row['Predicted_Delay_Minutes'] >= 0 else pd.NaT,
                        axis=1
                    )


                    # --- Rule-Based Diversion Suggestion ---
                    results['Recommendation'] = 'On Schedule (Adjusted Time)' # Default
                    diversion_threshold = 120 # Example: Suggest diversion if delay > 2 hours
                    potential_alternatives = [ap for ap in available_airports if ap != selected_airport and ap in airport_capacities]

                    results['Reason'] = '-' # Initialize Reason column
                    for idx in results.index: # Iterate using index
                        suggest_diversion = False
                        reason = ""
                        if accident_status == 1:
                            suggest_diversion = True
                            reason = "Runway Accident"
                        elif current_weather in BAD_WEATHER: # Check for bad weather condition
                            suggest_diversion = True
                            reason = f"Bad Weather Condition: {current_weather}"
                        # Check if Predicted_Delay_Minutes exists and is valid before comparison
                        elif 'Predicted_Delay_Minutes' in results.columns and pd.notna(results.loc[idx, 'Predicted_Delay_Minutes']) and results.loc[idx, 'Predicted_Delay_Minutes'] > diversion_threshold:
                            suggest_diversion = True
                            reason = f"High Predicted Delay ({results.loc[idx, 'Predicted_Delay_Minutes']} min)"

                        if suggest_diversion:
                            # Simple alternative suggestion
                            alt_suggestion = "Consider Diversion"
                            if potential_alternatives:
                                # Basic: Suggest first few alternatives (improve with capacity check/distance)
                                alt_suggestion += f" (e.g., {', '.join(potential_alternatives[:2])})"
                            else:
                                alt_suggestion += " (No alternatives found in data)"
                            results.loc[idx, 'Recommendation'] = alt_suggestion
                            results.loc[idx, 'Adjusted_DateTime'] = pd.NaT # No adjusted time if diverting
                            results.loc[idx, 'Reason'] = reason


                    # Prepare display format
                    results_display = results[['Flight_ID', 'Flight_Type', 'Airport', 'Scheduled_DateTime', 'Delayed_Time', 'Predicted_Delay_Minutes', 'Adjusted_DateTime', 'Recommendation', 'Reason']].copy()
                    results_display.rename(columns={'Delayed_Time': 'Original_Delay_Info'}, inplace=True) # Rename original delay column as it wasn't minutes

                    # Format datetime columns safely
                    results_display['Scheduled_DateTime'] = pd.to_datetime(results_display['Scheduled_DateTime']).dt.strftime('%Y-%m-%d %H:%M')
                    results_display['Adjusted_DateTime'] = pd.to_datetime(results_display['Adjusted_DateTime']).dt.strftime('%Y-%m-%d %H:%M')
                    results_display.fillna({'Reason': '-', 'Adjusted_DateTime': 'N/A', 'Predicted_Delay_Minutes': 'Error'}, inplace=True) # Clean up display


                    # Store results in session state
                    st.session_state.recommendations.append({
                        "input_airport": selected_airport,
                        "input_weather": current_weather,
                        "input_accident": accident_occurred,
                        "results_df": results_display
                    })

                except Exception as e:
                    st.error(f"An error occurred during prediction or recommendation generation: {e}")
                    import traceback
                    st.error("Traceback:")
                    st.code(traceback.format_exc())
            else:
                st.warning("No valid flights could be processed for prediction based on the provided data and inputs.")


    # --- Display Recommendations ---
    if 'recommendations' in st.session_state and st.session_state.recommendations:
        st.subheader("Generated Recommendations History")
        # Display recommendations in reverse chronological order (newest first)
        for i, rec in enumerate(reversed(st.session_state.recommendations)):
            expander_label = f"Scenario {len(st.session_state.recommendations) - i}: {rec['input_airport']} | Weather: {rec['input_weather']} | Accident: {rec['input_accident']}"
            with st.expander(expander_label, expanded=(i == 0)): # Expand the latest one by default
                st.dataframe(rec['results_df'], use_container_width=True)
                # Add download button for the specific recommendation
                csv = rec['results_df'].to_csv(index=False).encode('utf-8')
                st.download_button(
                    label=f"Download Scenario {len(st.session_state.recommendations) - i} as CSV",
                    data=csv,
                    file_name=f'recommendation_scenario_{len(st.session_state.recommendations) - i}.csv',
                    mime='text/csv',
                    key=f'download_btn_{i}' # Unique key for each button
                )

    elif submit_button: # Only show if submit was pressed but no recommendations generated
        st.info("No recommendations were generated. Check warnings above.")
    else:
        st.info("Submit conditions above to generate the first recommendation.")


else:
    # Error message if artifacts or data failed to load
    st.error("Application initialization failed. Please check the console/log for artifact loading errors and ensure all necessary files (model, data, preprocessors) are present and correct.")