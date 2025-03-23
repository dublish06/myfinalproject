import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.losses import Huber
from tensorflow.keras.callbacks import EarlyStopping
import os


os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Load the dataset
data = pd.read_csv("C:/Users/sahil/OneDrive/Documents/inhouseproject2/synthetic_data.csv")  # Replace with actual path

# Convert 'Pass/Fail' statuses to binary values (0 or 1)
status_columns = ['Disk Status', 'Memory Status', 'Uptime Status', 'Chrony Status']
for col in status_columns:
    data[col] = data[col].apply(lambda x: 1 if x == 'Pass' else 0)

# Select features and target variable
features = ['Disk Utilization (%)', 'Total Memory', 'Used Memory', 'Free Memory', 'Uptime (min)']
target = 'Used Memory'

# Normalize features and target separately
feature_scaler = StandardScaler()
target_scaler = MinMaxScaler()

data_scaled = feature_scaler.fit_transform(data[features])
target_scaled = target_scaler.fit_transform(data[[target]])  # Normalize only 'Used Memory'

# Prepare sequences for LSTM (Time Series Format)
sequence_length = 20  # Increased sequence length for better pattern recognition
X, y = [], []

for i in range(len(data_scaled) - sequence_length):
    X.append(data_scaled[i:i + sequence_length])  # 20 previous time steps as input
    y.append(target_scaled[i + sequence_length])  # Predict scaled 'Used Memory'

X, y = np.array(X), np.array(y)

# Train-Test split (80% training, 20% testing, shuffle enabled)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

# Build Optimized LSTM Model
model = Sequential([
    Input(shape=(sequence_length, len(features))),
    LSTM(64, activation='tanh', return_sequences=True),
    Dropout(0.2),
    LSTM(32, activation='tanh'),
    Dense(1)
])

model.compile(optimizer='adam', loss=Huber(delta=1.0))

model.summary()

# Add Early Stopping to Prevent Overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model
history = model.fit(X_train, y_train, epochs=100, batch_size=8, validation_data=(X_test, y_test),
                    verbose=1, callbacks=[early_stopping])

# Make predictions
y_pred = model.predict(X_test)

# Inverse Transform Predictions
y_test_actual = target_scaler.inverse_transform(y_test)
y_pred_actual = target_scaler.inverse_transform(y_pred)

# Plot Predictions vs Actual
plt.figure(figsize=(10, 5))
plt.plot(y_test_actual, label="Actual Usage", marker='o')
plt.plot(y_pred_actual, label="Predicted Usage", linestyle='dashed', marker='x')
plt.xlabel("Sample Index")
plt.ylabel("Used Memory")
plt.title("LSTM Model: Actual vs Predicted Memory Usage")
plt.legend()
plt.show()

# Compute RMSE for model evaluation
from sklearn.metrics import mean_squared_error

rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred_actual))
print(f"RMSE: {rmse:.2f} MB")

# Future Prediction (Next Time Step)
future_input = data_scaled[-sequence_length:]  # Last 'sequence_length' time steps
future_input = np.expand_dims(future_input, axis=0)  # Add batch dimension
future_prediction = model.predict(future_input)

# Inverse Transform Future Prediction
future_memory_usage = target_scaler.inverse_transform(future_prediction)

print(f"Predicted Future Used Memory: {future_memory_usage[0][0]:.2f} MB")
