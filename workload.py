import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, mean_squared_error, r2_score

# Load the CSV data
df = pd.read_csv("C:/Users/sahil/OneDrive/Documents/inhouseproject2/synthetic_data.csv")

# Feature selection
features = ['Disk Utilization (%)', 'Total Memory', 'Used Memory', 'Free Memory', 'Uptime (min)']
target = 'Overloaded'  # Add a target column based on thresholds

# Automatically create the target column
# Define an overloaded VM (example: high disk usage and low free memory)
df['Overloaded'] = ((df['Disk Utilization (%)'] > 70) | (df['Free Memory'] < 500)).astype(int)
df['Underworked'] = ((df['Disk Utilization (%)'] < 30) & (df['Free Memory'] > 1500)).astype(int)
# Split data into features and target
X = df[features]
y = df['Overloaded']

# Normalize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train a classifier
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)
importance = clf.feature_importances_
indices = np.argsort(importance)[::-1]

plt.figure(figsize=(10, 6))
plt.title("Feature Importances")
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


# Predict on all machines
df['Predicted_Overloaded'] = clf.predict(X_scaled)

# Identify overloaded machines
overloaded_vms = df[df['Predicted_Overloaded'] == 1]
underworked_vms = df[df['Underworked'] == 1]
print("Overloaded VMs:\n", overloaded_vms[['IP', 'Disk Utilization (%)', 'Free Memory']])
print("\nUnderworked VMs:\n", underworked_vms[['IP', 'Disk Utilization (%)', 'Free Memory']])
# Save the results to a new CSV
df.to_csv("vm_analysis_with_predictions.csv", index=False)
print("Results saved to 'vm_analysis_with_predictions.csv'.")
if not overloaded_vms.empty and not underworked_vms.empty:
    print("\nWorkload Redistribution Suggestions:")
    for index, overloaded_vm in overloaded_vms.iterrows():
        # Suggest the first underworked VM for redistribution
        underworked_vm = underworked_vms.iloc[0]
        print(f"Distribute workload of {overloaded_vm['IP']} to {underworked_vm['IP']}")
        # Remove the underworked VM from further suggestions
        underworked_vms = underworked_vms.iloc[1:]
        if underworked_vms.empty:
            break
else:
    print("\nNo sufficient data for workload redistribution.")

# Save the results to a new CSV
df.to_csv("vm_analysis_with_workload_suggestions.csv", index=False)
print("Results saved to 'vm_analysis_with_workload_suggestions.csv'.")