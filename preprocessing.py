import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
import joblib 
import numpy as np
import os
from datetime import time 

INPUT_EXCEL_FILE = 'air_traffic_dataset_with_weather.xlsx' 
PREPROCESSED_CSV_FILE = 'preprocessed_flights.csv'
ENCODER_FILE = 'onehot_encoder.joblib'
SCALER_FILE = 'standard_scaler.joblib'
TARGET_SCALER_FILE = 'target_scaler.joblib' 
TEST_SIZE = 0.2
VAL_SIZE = 0.1 
RANDOM_STATE = 42

try:
    df = pd.read_excel(INPUT_EXCEL_FILE)
    print(f"Successfully loaded data from {INPUT_EXCEL_FILE}")
    print(f"Initial data shape: {df.shape}")
    print("Initial columns:", df.columns.tolist())
    print("\nSample data:\n", df.head())
except FileNotFoundError:
    print(f"Error: Input file '{INPUT_EXCEL_FILE}' not found.")
    print("Please ensure the Excel file is in the same directory or provide the correct path.")
    exit()
except Exception as e:
    print(f"Error loading Excel file: {e}")
    exit()

print("\nStarting Feature Engineering...")

try:
    datetime_str = df['Date'].astype(str) + ' ' + df['Time'].astype(str)
    df['Scheduled_DateTime'] = pd.to_datetime(datetime_str, errors='coerce')
except Exception as e:
    print(f"Initial datetime parsing failed: {e}. Trying alternative formats.")
    try:
        datetime_str = df['Date'].astype(str) + ' ' + df['Time'].astype(str)
        df['Scheduled_DateTime'] = pd.to_datetime(datetime_str, errors='coerce', infer_datetime_format=True)
    except Exception as e2:
        print(f"Alternative datetime parsing also failed: {e2}. Check Date/Time columns.")

initial_rows = df.shape[0]
df.dropna(subset=['Scheduled_DateTime'], inplace=True)
rows_after_dropna = df.shape[0]
if initial_rows != rows_after_dropna:
    print(f"Dropped {initial_rows - rows_after_dropna} rows due to invalid Date/Time format.")
print(f"Shape after datetime conversion and dropping NaNs: {df.shape}")

df['Hour'] = df['Scheduled_DateTime'].dt.hour
df['Minute'] = df['Scheduled_DateTime'].dt.minute
df['DayOfWeek'] = df['Scheduled_DateTime'].dt.dayofweek 
df['Month'] = df['Scheduled_DateTime'].dt.month
df['DayOfYear'] = df['Scheduled_DateTime'].dt.dayofyear

print("Extracted time features: Hour, Minute, DayOfWeek, Month, DayOfYear")

print("Processing 'Delayed_Time' column...")

def convert_delay_to_minutes(delay_value):
    if pd.isna(delay_value):
        return 0 
    if isinstance(delay_value, (int, float)):
        return int(delay_value)
    try:
        if isinstance(delay_value, time):
            return delay_value.hour * 60 + delay_value.minute
        elif isinstance(delay_value, str) and ':' in delay_value:
            parts = delay_value.split(':')
            if len(parts) == 2:
                try:
                    hours = int(parts[0])
                    minutes = int(parts[1])
                    return hours * 60 + minutes
                except ValueError:
                    print(f"Warning: Could not parse numeric parts from delay string '{delay_value}'. Treating as 0 delay.")
                    return 0
            else:
                print(f"Warning: Invalid HH:MM format '{delay_value}'. Treating as 0 delay.")
                return 0 
        else:
            print(f"Warning: Unexpected delay value type '{type(delay_value)}' ('{delay_value}'). Attempting direct int conversion.")
            return int(delay_value)
    except (ValueError, TypeError) as e:
        print(f"Warning: Error parsing delay value '{delay_value}' ({e}). Treating as 0 delay.")
        return 0

df['Delayed_Time_Minutes'] = df['Delayed_Time'].apply(convert_delay_to_minutes)

print("Converted 'Delayed_Time' to 'Delayed_Time_Minutes'.")
print("Sample of calculated delay minutes:\n", df[['Flight_ID', 'Delayed_Time', 'Delayed_Time_Minutes']].head())

df['Is_Delayed'] = (df['Delayed_Time_Minutes'] > 0).astype(int)
print("Created 'Is_Delayed' based on minutes.")

categorical_features = ['Flight_Type', 'City', 'Airport', 'Weather_Condition']
numerical_features = ['Airport_Capacity', 'Hour', 'Minute', 'DayOfWeek', 'Month', 'DayOfYear']
target_variable = 'Delayed_Time_Minutes' 

all_required_cols = ['Flight_ID'] + categorical_features + numerical_features + [target_variable]
missing_cols = [col for col in all_required_cols if col not in df.columns]
if missing_cols:
    print(f"Error: Missing required columns after initial processing: {missing_cols}")
    print(f"Available columns: {df.columns.tolist()}")
    exit()

df_processed = df[all_required_cols + ['Is_Delayed', 'Scheduled_DateTime']].copy()
print(f"Selected columns for processing: {all_required_cols}")
print(f"Shape of df_processed before encoding/scaling: {df_processed.shape}") 

if df_processed.empty:
    print("CRITICAL Error: df_processed DataFrame is empty before encoding/scaling steps.")
    print("This might indicate issues in data loading, datetime conversion, or delay parsing.")
    exit()

print("Encoding categorical features using OneHotEncoder...")
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore') 

try:
    encoded_cats = encoder.fit_transform(df_processed[categorical_features])
    encoded_cat_df = pd.DataFrame(encoded_cats, columns=encoder.get_feature_names_out(categorical_features), index=df_processed.index)
except Exception as e:
    print(f"Error during OneHotEncoder fit_transform: {e}")
    print("Check the content of categorical columns:", df_processed[categorical_features].info())
    exit()

joblib.dump(encoder, ENCODER_FILE)
print(f"Saved OneHotEncoder to {ENCODER_FILE}")

print("Scaling numerical features using StandardScaler...")
scaler = StandardScaler()
try:
    scaled_nums = scaler.fit_transform(df_processed[numerical_features])
    scaled_num_df = pd.DataFrame(scaled_nums, columns=numerical_features, index=df_processed.index)
except Exception as e:
    print(f"Error during StandardScaler fit_transform: {e}")
    print("Check the content of numerical columns:", df_processed[numerical_features].info())
    exit()

joblib.dump(scaler, SCALER_FILE)
print(f"Saved StandardScaler to {SCALER_FILE}")

print(f"Scaling target variable ({target_variable}) using MinMaxScaler...")
target_scaler = MinMaxScaler()
try:
    scaled_target = target_scaler.fit_transform(df_processed[[target_variable]])
    scaled_target_df = pd.DataFrame(scaled_target, columns=[target_variable + '_scaled'], index=df_processed.index)
except Exception as e:
    print(f"Error during MinMaxScaler fit_transform on target variable: {e}")
    print("Check the content of the target column:", df_processed[[target_variable]].info())
    exit()

joblib.dump(target_scaler, TARGET_SCALER_FILE)
print(f"Saved Target MinMaxScaler to {TARGET_SCALER_FILE}")

print("Combining processed features into the final DataFrame...")
df_final = pd.concat([
    df_processed[['Flight_ID', 'Scheduled_DateTime']], 
    scaled_num_df, 
    encoded_cat_df, 
    scaled_target_df, 
    df_processed[['Is_Delayed', target_variable]] 
], axis=1)

df_final.rename(columns={
    target_variable + '_scaled': 'Target_Scaled_Delay', 
    target_variable: 'Original_Delayed_Time_Minutes' 
}, inplace=True)

print("Combined all processed features.")
print(f"Final preprocessed data shape: {df_final.shape}")
print("Final columns:", df_final.columns.tolist())
print("\nSample preprocessed data:\n", df_final.head())

try:
    df_final.to_csv(PREPROCESSED_CSV_FILE, index=False)
    print(f"\nSuccessfully saved preprocessed data to {PREPROCESSED_CSV_FILE}")
except Exception as e:
    print(f"Error saving preprocessed data to CSV: {e}")

print("\nPreprocessing script finished.")