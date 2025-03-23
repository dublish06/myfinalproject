import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, mean_squared_error, r2_score

# Load the CSV data
df = pd.read_csv("C:/Users/sahil/OneDrive/Documents/inhouseproject2/synthetic_data.csv")

# Feature selection
features = ['Disk Utilization (%)', 'Total Memory', 'Used Memory', 'Free Memory', 'Uptime (min)']

# Define a new target column for predicting failures
df['About_to_Fail'] = (
    (df['Disk Utilization (%)'] > 85) |  # High disk usage
    (df['Free Memory'] < 300) |          # Very low free memory
    (df['Uptime (min)'] > df['Uptime (min)'].quantile(0.90))  # Long-running servers
).astype(int)

# Split data into features and target
X = df[features]
y = df['About_to_Fail']

# Normalize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train a classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Feature importance plot
importance = clf.feature_importances_
indices = np.argsort(importance)[::-1]

plt.figure(figsize=(10, 6))
plt.title("Feature Importances for Failure Prediction")
plt.bar(range(len(features)), importance[indices], align="center")
plt.xticks(range(len(features)), [features[i] for i in indices], rotation=90)
plt.xlabel("Features")
plt.ylabel("Importance Score")
plt.tight_layout()
plt.show()

# Evaluate the model
y_pred = clf.predict(X_test)
print("Classification Report:\n", classification_report(y_test, y_pred))
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nMean Squared Error:", mse)
print("R² Score:", r2)

# Predict failures on all machines
df['Predicted_Failure'] = clf.predict(X_scaled)

# Identify at-risk machines
at_risk_vms = df[df['Predicted_Failure'] == 1]
print("Machines at Risk of Failure:\n", at_risk_vms[['IP', 'Disk Utilization (%)', 'Free Memory']])

# Alert users about resource allocation
if not at_risk_vms.empty:
    print("\n⚠️ ALERT: The following machines need more resources to prevent failure ⚠️")
    for index, vm in at_risk_vms.iterrows():
        print(f"⚠️ Machine {vm['IP']} is at risk. Consider adding more memory or balancing its workload.")

# Save results
df.to_csv("vm_failure_prediction.csv", index=False)
print("Results saved to 'vm_failure_prediction.csv'.")
