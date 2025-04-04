import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
import joblib
import math

PREPROCESSED_CSV_FILE = 'preprocessed_flights.csv'
MODEL_SAVE_PATH = 'lstm_rescheduler_model.h5'
TARGET_SCALER_FILE = 'target_scaler.joblib'

LSTM_UNITS = 64
DROPOUT_RATE = 0.2
EPOCHS = 50
BATCH_SIZE = 32
VAL_SPLIT = 0.2
TEST_SIZE_SPLIT = 0.15
RANDOM_STATE = 42

try:
    df = pd.read_csv(PREPROCESSED_CSV_FILE)
    print(f"Successfully loaded preprocessed data from {PREPROCESSED_CSV_FILE}")
    print(f"Data shape: {df.shape}")
except FileNotFoundError:
    print(f"Error: Preprocessed file '{PREPROCESSED_CSV_FILE}' not found.")
    print("Please run the preprocessing script first.")
    exit()
except Exception as e:
    print(f"Error loading preprocessed data: {e}")
    exit()

print("\nPreparing data for LSTM training...")

feature_columns = [col for col in df.columns if col not in ['Flight_ID', 'Scheduled_DateTime', 'Target_Scaled_Delay', 'Is_Delayed', 'Original_Delayed_Time_Minutes']]
target_column = 'Target_Scaled_Delay'

X = df[feature_columns].values
y = df[target_column].values

if X.shape[0] == 0 or y.shape[0] == 0:
    print("Error: Feature matrix X or target vector y is empty after loading/selecting columns.")
    print(f"X shape: {X.shape}, y shape: {y.shape}")
    exit()

X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=TEST_SIZE_SPLIT, random_state=RANDOM_STATE
)

if X_test.shape[0] == 0 or y_test.shape[0] == 0:
     print("Error: Test set is empty after train/test split.")
     print(f"Original X shape: {X.shape}, Test size: {TEST_SIZE_SPLIT}")
     exit()

X_train_val_lstm = np.reshape(X_train_val, (X_train_val.shape[0], 1, X_train_val.shape[1]))
X_test_lstm = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

print(f"Data shapes for LSTM:")
print(f"X_train_val_lstm: {X_train_val_lstm.shape}")
print(f"y_train_val: {y_train_val.shape}")
print(f"X_test_lstm: {X_test_lstm.shape}")
print(f"y_test (scaled): {y_test.shape}")

print("\nBuilding LSTM model...")
n_features = X_train_val_lstm.shape[2]

model = Sequential([
    Input(shape=(1, n_features)),
    LSTM(LSTM_UNITS, return_sequences=False),
    Dropout(DROPOUT_RATE),
    Dense(LSTM_UNITS // 2, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

model.summary()

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)
model_checkpoint = ModelCheckpoint(MODEL_SAVE_PATH, monitor='val_loss', save_best_only=True, verbose=1)

print("\nStarting model training...")
history = model.fit(
    X_train_val_lstm, y_train_val,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=VAL_SPLIT,
    callbacks=[early_stopping, model_checkpoint],
    verbose=1
)

print("\nModel training finished.")
print(f"Best model saved to: {MODEL_SAVE_PATH}")

print("\n--- Evaluating Model Performance on Test Set ---")
try:
    best_model = tf.keras.models.load_model(MODEL_SAVE_PATH)

    test_loss_scaled, test_mae_scaled = best_model.evaluate(X_test_lstm, y_test, verbose=0)
    print(f"\nPerformance on Scaled Data (0-1 range):")
    print(f"  Test Loss (MSE): {test_loss_scaled:.5f}")
    print(f"  Test Mean Absolute Error (MAE): {test_mae_scaled:.5f}")

    y_pred_scaled = best_model.predict(X_test_lstm)

    try:
        target_scaler = joblib.load(TARGET_SCALER_FILE)

        if y_test.ndim == 1:
            y_test = y_test.reshape(-1, 1)

        y_pred_original = target_scaler.inverse_transform(y_pred_scaled)
        y_test_original = target_scaler.inverse_transform(y_test)

        y_test_original_flat = y_test_original.flatten()
        y_pred_original_flat = y_pred_original.flatten()

        print(f"\nPerformance on Original Scale (Delay Minutes):")

        mae_original = mean_absolute_error(y_test_original_flat, y_pred_original_flat)
        print(f"  Mean Absolute Error (MAE): {mae_original:.2f} minutes")
        print(f"  (Average absolute difference between predicted and actual delay)")

        mse_original = mean_squared_error(y_test_original_flat, y_pred_original_flat)
        print(f"  Mean Squared Error (MSE): {mse_original:.2f}")

        rmse_original = math.sqrt(mse_original)
        print(f"  Root Mean Squared Error (RMSE): {rmse_original:.2f} minutes")
        print(f"  (Standard deviation of the prediction errors)")

        r2 = r2_score(y_test_original_flat, y_pred_original_flat)
        print(f"  R-squared (R²): {r2:.4f}")
        print(f"  (Proportion of variance explained by the model. Closer to 1 is better)")

        mask = y_test_original_flat != 0
        if np.any(mask):
            mape = mean_absolute_percentage_error(y_test_original_flat[mask], y_pred_original_flat[mask]) * 100
            print(f"  Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
            print(f"  (Average percentage difference. Calculated excluding zero actual delays)")
        else:
            print("  MAPE: Cannot be calculated as all actual values in the test set are zero.")

        print("\nNote: Accuracy is typically used for classification.")
        print("For this regression task, MAE, RMSE, and R² are key performance indicators.")

    except FileNotFoundError:
        print(f"\nWarning: Target scaler file '{TARGET_SCALER_FILE}' not found.")
        print("Cannot calculate metrics in the original scale (minutes).")
    except Exception as e:
        print(f"\nError during inverse transform or metric calculation: {e}")

except Exception as e:
    print(f"Error loading the saved model from '{MODEL_SAVE_PATH}': {e}")

print("\nModel training and evaluation script finished.")
