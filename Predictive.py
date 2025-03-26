import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, PolynomialFeatures
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_absolute_error
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from imblearn.over_sampling import SMOTE

# Load the dataset
file_path = "C:/Users/sahil/OneDrive/Documents/inhouseproject2/synthetic_data.csv"  # Update path if needed
df = pd.read_csv(file_path)

# Convert categorical "Pass"/"Fail" to numerical (1 = Pass, 0 = Fail)
status_columns = ["Disk Status", "Memory Status", "Uptime Status", "Chrony Status"]
for col in status_columns:
    df[col] = df[col].apply(lambda x: 1 if x == "Pass" else 0)

# Handle missing values
numeric_cols = df.select_dtypes(include=['number']).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

# Encode IP addresses as categorical values
if 'IP' in df.columns:
    df["IP"] = LabelEncoder().fit_transform(df["IP"].astype(str))

# Select features for failure prediction
features = ["IP", "Disk Utilization (%)", "Total Memory", "Used Memory", "Free Memory", "Uptime (min)"]
target = "Disk Status"

X = df[features]
y = df[target]

# Check for class imbalance
class_counts = y.value_counts()
print("🔹 Class Distribution Before SMOTE:\n", class_counts)

# Train-test split for failure prediction
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Apply SMOTE to balance the dataset
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Check class balance after SMOTE
print("🔹 Class Distribution After SMOTE:\n", pd.Series(y_train_resampled).value_counts())

# Calculate scale_pos_weight for XGBoost
num_neg, num_pos = np.bincount(y_train)
scale_pos_weight = num_neg / num_pos

# ========================================
# Hyperparameter tuning using RandomizedSearchCV for XGBoost
param_dist = {
    'max_depth': [3, 5, 7, 9],
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [100, 200, 300],
    'subsample': [0.7, 0.8, 1.0],
    'colsample_bytree': [0.7, 0.8, 1.0],
    'gamma': [0, 0.1, 0.2],
    'scale_pos_weight': [1, 2]
}

xgb_model = XGBClassifier()

random_search = RandomizedSearchCV(estimator=xgb_model, param_distributions=param_dist, n_iter=10,
                                   scoring='accuracy', cv=5, random_state=42, n_jobs=-1)

random_search.fit(X_train_resampled, y_train_resampled)

# Best parameters
print("🔹 Best XGBoost Parameters from RandomizedSearchCV:", random_search.best_params_)

# Using the best model
best_model = random_search.best_estimator_
y_pred = best_model.predict(X_test)

# Evaluate the performance of XGBoost
print("🔹 Failure Prediction Accuracy (XGBoost):", accuracy_score(y_test, y_pred))
print("🔹 Classification Report (XGBoost):\n", classification_report(y_test, y_pred, zero_division=1))
print("🔹 Confusion Matrix (XGBoost):\n", confusion_matrix(y_test, y_pred))

# ========================================
# Alternative Models (Random Forest and Logistic Regression)
# ========================================

# Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_resampled, y_train_resampled)
y_pred_rf = rf_model.predict(X_test)

# Evaluate Random Forest
print("🔹 Failure Prediction Accuracy (Random Forest):", accuracy_score(y_test, y_pred_rf))
print("🔹 Classification Report (Random Forest):\n", classification_report(y_test, y_pred_rf))
print("🔹 Confusion Matrix (Random Forest):\n", confusion_matrix(y_test, y_pred_rf))

# Logistic Regression
logreg_model = LogisticRegression(random_state=42,solver='saga',max_iter=1000  )
logreg_model.fit(X_train_resampled, y_train_resampled)
y_pred_logreg = logreg_model.predict(X_test)

# Evaluate Logistic Regression
print("🔹 Failure Prediction Accuracy (Logistic Regression):", accuracy_score(y_test, y_pred_logreg))
print("🔹 Classification Report (Logistic Regression):\n", classification_report(y_test, y_pred_logreg))
print("🔹 Confusion Matrix (Logistic Regression):\n", confusion_matrix(y_test, y_pred_logreg))

# ========================================
# Feature Importance Plot for XGBoost
plt.figure(figsize=(8, 5))
sns.barplot(x=best_model.feature_importances_, y=features)
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Feature Importance in Predicting Disk Failures")
plt.show()

# ==============================
# 📊 Predictive Analytics for Resource Management
# ==============================

# Selecting features for resource forecasting
resource_features = ["Disk Utilization (%)", "Used Memory", "Uptime (min)"]

# Simulating timestamps
df["Time Index"] = range(len(df))

# Forecast future disk utilization, memory usage, and uptime
for feature in resource_features:
    X_resource = df[["Time Index"]]
    y_resource = df[feature]

    # Splitting data for training
    X_train, X_test, y_train, y_test = train_test_split(X_resource, y_resource, test_size=0.2, random_state=42)

    # Training a Linear Regression model for forecasting
    resource_model = LinearRegression()
    resource_model.fit(X_train, y_train)

    # Predict future values
    y_pred_resource = resource_model.predict(X_test)

    # Evaluate prediction
    mae = mean_absolute_error(y_test, y_pred_resource)
    print(f"📊 Mean Absolute Error for {feature} prediction: {mae:.2f}")