import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv("E:/aishwaryad_aiml/company_sales_data.csv")

# Create a figure with subplots
fig, ax = plt.subplots(2, 1, figsize=(10, 8))

# Subplot 1: Bathing Soap Sales
ax[0].plot(df["month_number"], df["bathingsoap"], marker='o', linestyle='-', color='b', label="Bathing Soap Sales")
ax[0].set_title("Bathing Soap Sales per Month")
ax[0].set_xlabel("Month Number")
ax[0].set_ylabel("Units Sold")
ax[0].grid(True)
ax[0].legend()

# Subplot 2: Facewash Sales
ax[1].plot(df["month_number"], df["facewash"], marker='s', linestyle='--', color='g', label="Facewash Sales")
ax[1].set_title("Facewash Sales per Month")
ax[1].set_xlabel("Month Number")
ax[1].set_ylabel("Units Sold")
ax[1].grid(True)
ax[1].legend()

# Adjust layout and display
plt.tight_layout()
plt.show()
