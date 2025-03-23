import os
import glob
import pandas as pd
import re
import numpy as np
from faker import Faker
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import matplotlib.pyplot as plt
# Specify the dirctory where files are generated
directory = "C:/Users/sahil/OneDrive/Documents/inhouseproject2"
file_path = 'C:/Users/sahil/OneDrive/Documents/inhouseproject2/synthetic_data.csv'
data = pd.read_csv(file_path)
label_encoders = {}
categorical_columns = ['Disk Status', 'Memory Status', 'Uptime Status', 'Chrony Status']

# Get the most recent Excel and text files
latest_excel_file = max(glob.glob(f"{directory}/*.xlsx"), key=os.path.getctime)
latest_text_file = max(glob.glob(f"{directory}/*.txt"), key=os.path.getctime)

print("Latest Excel file:", latest_excel_file)
print("Latest Text file:", latest_text_file)
# Load the text file
with open(latest_text_file, "r") as file:
    lines = file.readlines()

# Initialize variables for parsing
data = []
current_ip = None
current_metrics = {}

# Iterate through lines
for line in lines:
    line = line.strip()

    # Extract IP Address
    if "Healthcheck status" in line:
        if current_ip and current_metrics:
            data.append({"IP": current_ip, **current_metrics})
        current_ip = line.split()[-1]
        current_metrics = {}

    # Extract Disk Utilization
    if line.startswith("/dev/mapper"):
        utilization = int(line.split()[4].strip('%'))
        current_metrics["Disk Utilization (%)"] = utilization

    # Extract Memory Info
    if line.startswith("Mem:"):
        memory_info = line.split()
        current_metrics["Total Memory"] = int(memory_info[1])
        current_metrics["Used Memory"] = int(memory_info[2])
        current_metrics["Free Memory"] = int(memory_info[3])

    # Extract Uptime Info
    if "up" in line and "min" in line:
        uptime = re.search(r'up (\d+) min', line)
        if uptime:
            current_metrics["Uptime (min)"] = int(uptime.group(1))

# Append the last IP's data
if current_ip and current_metrics:
    data.append({"IP": current_ip, **current_metrics})

# Convert to DataFrame
df_file1 = pd.DataFrame(data)
# Load the Excel file
df_file2 = pd.read_excel(latest_excel_file)

# Rename columns for consistency
df_file2.rename(columns={
    "Disk Utilization Status": "Disk Status",
    "Available Memory Status": "Memory Status",
    "Uptime Status": "Uptime Status",
    "Chrony Status": "Chrony Status"
}, inplace=True)
# Merge datasets on IP
df_merged = pd.merge(df_file1, df_file2, on="IP", how="inner")
# Save the combined dataset
df_merged.to_csv("merged_dataset.csv", index=False)
print(df_merged)
faker = Faker()

# Parameters for synthetic data
num_rows = 100  # Number of synthetic rows

# Generate synthetic data
data = []
for _ in range(num_rows):
    ip = f"10.0.0.{np.random.randint(1, 255)}"  # Random IP addresses
    disk_utilization = np.random.randint(1, 100)  # Disk utilization percentage
    total_memory = np.random.choice([1024, 2048, 4096, 8192])  # Memory in MB
    used_memory = np.random.randint(1, total_memory)  # Used memory less than total
    free_memory = total_memory - used_memory
    uptime = np.random.randint(1, 1440)  # Uptime in minutes (up to 24 hours)
    disk_status = np.random.choice(["Pass", "Fail"], p=[0.8, 0.2])  # 80% Pass
    memory_status = np.random.choice(["Pass", "Fail"], p=[0.7, 0.3])  # 70% Pass
    uptime_status = np.random.choice(["Pass", "Fail"], p=[0.9, 0.1])  # 90% Pass
    chrony_status = np.random.choice(["Pass", "Fail"], p=[0.85, 0.15])  # 85% Pass

    data.append({
        "IP": ip,
        "Disk Utilization (%)": disk_utilization,
        "Total Memory": total_memory,
        "Used Memory": used_memory,
        "Free Memory": free_memory,
        "Uptime (min)": uptime,
        "Disk Status": disk_status,
        "Memory Status": memory_status,
        "Uptime Status": uptime_status,
        "Chrony Status": chrony_status
    })

# Create DataFrame
synthetic_df = pd.DataFrame(data)

# Shuffle data to ensure randomness
synthetic_df = shuffle(synthetic_df).reset_index(drop=True)

# Save to CSV
synthetic_df.to_csv("synthetic_data.csv", index=False)


print("Synthetic data generated successfully!")
# Ensure 'data' is a DataFrame (rename variable to avoid conflict)
# Ensure 'data' is a DataFrame
df = pd.DataFrame(data) if isinstance(data, list) else data

# Encode categorical columns
categorical_columns = ['Disk Status', 'Memory Status', 'Uptime Status', 'Chrony Status']
label_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Split features and target
X = df.drop(columns=['IP', 'Disk Utilization (%)'])  # Drop 'IP' and target column from features
y = df['Disk Utilization (%)']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R^2 Score:", r2)
# Change target to Free Memory
# Ensure 'data' is a DataFrame
if isinstance(data, list):
    df = pd.DataFrame(data)  # Convert list of dictionaries to DataFrame
else:
    df = data  # If already a DataFrame, use as is

# Access the target column
y = df['Free Memory']  # Replace target variable
# Replace target variable
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train and evaluate a new model for Free Memory
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Free Memory Prediction - Mean Squared Error:", mean_squared_error(y_test, y_pred))
# Example for Workload Prediction
# Ensure 'data' is a DataFrame
if isinstance(data, list):
    df = pd.DataFrame(data)  # Convert list of dictionaries to DataFrame
else:
    df = data  # If already a DataFrame, use as is

# Now safely access the column
y = df['Uptime (min)']  # Predicting uptime as a proxy for workload
# Predicting uptime as a proxy for workload
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Workload Prediction - Mean Squared Error:", mean_squared_error(y_test, y_pred))


# Save model
joblib.dump(model, 'workload_predictor.pkl')

# Load model later
loaded_model = joblib.load('workload_predictor.pkl')


# Feature Importance
importance = model.feature_importances_
indices = np.argsort(importance)[::-1]

plt.figure(figsize=(10, 6))
plt.title("Feature Importances")
plt.bar(range(X.shape[1]), importance[indices], align="center")
plt.xticks(range(X.shape[1]), X.columns[indices], rotation=90)
plt.tight_layout()
plt.show()

vm_data = [10.0, 50, 1500, 800, 500, 30, 1, 1]  # Ensure this has exactly 8 features matching the scaler


def predict_overload(vm_data):
    # Align vm_data to the required feature set
    vm_data_aligned = vm_data[:8]  # Take only the first 8 features

    # Scale the input data
    vm_data_aligned = np.array(vm_data_aligned).reshape(1, -1)  # Convert it to 2D array for scaling (1 row, n columns)

    # Transform the data using the scaler
    vm_data_scaled = scaler.transform(vm_data_aligned)  # Now it should be a 2D array with 8 features

    # Use your trained model to predict
    prediction = model.predict(vm_data_scaled)  # This should call the predict method and pass the scaled input

    return prediction



# Function to distribute workload
def distribute_workload(vm_ip):
    print(f"VM with IP {vm_ip} is overloaded. Distributing workload...")


# Example VM data for testing the function
# Example new VM data (this should be a list or a 1D array)
new_vm_data = [10.0, 50, 1500, 800, 500, 30, 1, 0]  # Replace this with actual data for the new VM

# Predict if the new VM is overloaded
result = predict_overload(new_vm_data)
print(f"The VM is {result}")


# Inform the user about the VM overload
if result > 80:  # Now comparing the actual predicted value with 80
    # Assuming 80 as the overload threshold
    distribute_workload("10.0.0.1")  # Example VM IP to distribute work

else:
    print("This VM is operating normally.")






