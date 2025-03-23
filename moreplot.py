import numpy as np
import dash
from dash import dcc, html
import plotly.graph_objects as go
from flask import Flask
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve

# -------------------- Load & Preprocess Data --------------------
df = pd.read_csv("C:/Users/sahil/OneDrive/Documents/inhouseproject2/synthetic_data.csv")

# Rename columns to match expected names
df.rename(columns={
    "Disk Utilization (%)": "disk_utilization",
    "Total Memory": "total_memory",
    "Used Memory": "used_memory",
    "Free Memory": "free_memory",
    "Uptime (min)": "uptime",
    "Disk Status": "disk_status",
    "Memory Status": "memory_status",
    "Uptime Status": "uptime_status",
    "Chrony Status": "chrony_status"
}, inplace=True)
df['timestamp'] = pd.date_range(start='2024-01-01', periods=len(df), freq='min')  # Generates timestamps


# Convert necessary columns to numeric
df["disk_utilization"] = pd.to_numeric(df["disk_utilization"], errors="coerce")
df["uptime"] = pd.to_numeric(df["uptime"], errors="coerce")
df["used_memory"] = pd.to_numeric(df["used_memory"], errors="coerce")
df["free_memory"] = pd.to_numeric(df["free_memory"], errors="coerce")

# Feature engineering: Calculate memory utilization as a percentage
df["memory_utilization"] = (df["used_memory"] / df["total_memory"]) * 100

# Drop unnecessary categorical columns if not needed
df.drop(["disk_status", "memory_status", "uptime_status", "chrony_status"], axis=1, inplace=True)


df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
df.dropna(subset=['timestamp', 'memory_utilization', 'uptime'], inplace=True)
df.set_index('timestamp', inplace=True)
df = df.astype({'memory_utilization': 'float64', 'uptime': 'float64'})

# -------------------- Memory Utilization Forecasting --------------------
def forecast_memory_utilization(series):
    if len(series) > 10:
        model = ARIMA(series, order=(3, 1, 0))
        model_fit = model.fit()
        return model_fit.forecast(steps=10)
    return np.array([])

memory_forecast = forecast_memory_utilization(df['memory_utilization'])

# -------------------- Uptime Stability Analysis --------------------
iso_forest = IsolationForest(contamination=0.05, random_state=42)
df['uptime_anomaly'] = iso_forest.fit_predict(df[['uptime']])

# -------------------- Disk Utilization Prediction --------------------
X = df[['disk_utilization', 'memory_utilization']]
y = (df['disk_utilization'] > 80).astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)


rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
y_prob = rf_model.predict_proba(X_test)[:, 1]

# -------------------- Machine Health Scoring --------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
kmeans = KMeans(n_clusters=3, random_state=42, n_init='auto')
df['health_score'] = kmeans.fit_predict(X_scaled)

# -------------------- Flask App & Dashboard --------------------
server = Flask(__name__)
app = dash.Dash(__name__, server=server)
# -------------------- ROC Curve Calculation --------------------

# Ensure no NaN or infinite values in y_prob
print("NaN values in y_prob:", np.isnan(y_prob).sum())
print("Infinite values in y_prob:", np.isinf(y_prob).sum())

# Replace NaNs/Infs with 0
y_prob = np.nan_to_num(y_prob)

# Compute ROC curve
fpr, tpr, _ = roc_curve(y_test, y_prob)

app.layout = html.Div(children=[
    html.H1("🔹 Real-Time System Monitoring Dashboard"),
    dcc.Graph(id='memory-forecast', figure={
        'data': [go.Scatter(y=memory_forecast, mode='lines', name='Forecast')],
        'layout': go.Layout(title="Memory Utilization Forecast", xaxis_title="Time", yaxis_title="Memory %")
    }),
    dcc.Graph(id='uptime-anomalies', figure={
        'data': [
            go.Box(y=df['uptime'], name="Uptime"),
            go.Scatter(y=df[df['uptime_anomaly'] == -1]['uptime'], mode='markers', name="Anomalies", marker=dict(color='red'))
        ],
        'layout': go.Layout(title="Uptime Stability Analysis")
    }),
    dcc.Graph(id='disk-utilization', figure={
        'data': [go.Scatter(y=df['disk_utilization'], mode='lines', name="Disk Utilization")],
        'layout': go.Layout(title="Disk Utilization Over Time")
    }),
    dcc.Graph(id='health-score', figure={
        'data': [go.Heatmap(z=df['health_score'].values.reshape(-1, 1), colorscale='RdYlGn')],
        'layout': go.Layout(title="Machine Health Score Heatmap")
    }),

    dcc.Graph(id='roc-curve', figure={
         'data': [go.Scatter(x=fpr, y=tpr, mode='lines', name="ROC Curve")],
         'layout': go.Layout(title="Alert Prediction ROC Curve",
                        xaxis_title="False Positive Rate",
                        yaxis_title="True Positive Rate")
})


])

if __name__ == '__main__':
    app.run_server(debug=True, use_reloader=False, threaded=True)
