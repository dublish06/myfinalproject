import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

# Read data from CSV
df = pd.read_csv("C:/Users/sahil/OneDrive/Documents/inhouseproject2/synthetic_data.csv")  # Update path if needed

# Encode categorical columns
label_encoder = LabelEncoder()
df['Disk Status'] = label_encoder.fit_transform(df['Disk Status'])
df['Memory Status'] = label_encoder.fit_transform(df['Memory Status'])
df['Uptime Status'] = label_encoder.fit_transform(df['Uptime Status'])

# Descriptive Statistics
print(df.describe())

# Save visualizations instead of showing them
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='Used Memory', y='Free Memory', hue='Disk Utilization (%)', palette='coolwarm', s=100)
plt.title('Used Memory vs Free Memory with Disk Utilization')
plt.xlabel('Used Memory')
plt.ylabel('Free Memory')
plt.savefig("static/memory_vs_disk.png")  # Save graph
plt.close()

plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='Uptime Status', y='Disk Utilization (%)')
plt.title('Uptime Status vs Disk Utilization')
plt.xlabel('Uptime Status')
plt.ylabel('Disk Utilization (%)')
plt.savefig("static/uptime_vs_disk.png")  # Save graph
plt.close()

df_numeric = df.select_dtypes(include=['number'])
plt.figure(figsize=(10, 6))
sns.heatmap(df_numeric.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.savefig("static/correlation_heatmap.png")  # Save graph
plt.close()

print("Graphs saved successfully!")
