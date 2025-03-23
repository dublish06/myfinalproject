import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# Load CSV data from file
df = pd.read_csv("C:/Users/sahil/OneDrive/Documents/inhouseproject2/synthetic_data.csv")

# Convert memory usage to percentage
df["Memory Utilization (%)"] = (df["Used Memory"] / df["Total Memory"]) * 100

# Simulating past usage data (Ideally, you'd fetch real historical data)
np.random.seed(42)  # For reproducibility
df["Past Memory Usage (%)"] = df["Memory Utilization (%)"] - np.random.randint(1, 10, size=len(df))

# Predict future memory utilization using Linear Regression
future_time = 60  # Predicting 1 hour into the future (adjustable)

alerts = []

for index, row in df.iterrows():
    # Prepare data for prediction
    past_usage = np.array([row["Past Memory Usage (%)"], row["Memory Utilization (%)"]]).reshape(-1, 1)
    timestamps = np.array([0, future_time]).reshape(-1, 1)  # Time intervals (now and future)

    model = LinearRegression()
    model.fit(timestamps, past_usage)

    future_memory_util = model.predict([[future_time]])[0][0]  # Predict next hour usage

    # Check for memory threshold exceedance
    if future_memory_util > 80:
        alerts.append(f"🚨 ALERT: {row['IP']} is predicted to exceed 80% memory utilization in {future_time} mins!")

    # Detect system uptime instability
    if row["Uptime (min)"] < 500 and row["Uptime Status"] == "Fail":
        alerts.append(f"⚠️ WARNING: {row['IP']} shows unstable uptime patterns! ({row['Uptime (min)']} min)")

# Output alerts
for alert in alerts:
    print(alert)
