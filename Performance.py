import pandas as pd
import dash
from dash import dcc, html, Input, Output
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset once
df = pd.read_csv("C:/Users/sahil/OneDrive/Documents/inhouseproject2/synthetic_data.csv")

# Create Timestamp column
start_time = pd.to_datetime('2025-01-01 00:00')
df['Timestamp'] = start_time + pd.to_timedelta(df['Uptime (min)'], unit='m')

# Sort the DataFrame by Timestamp
df = df.sort_values(by="Timestamp")

# Prepare data for forecasting
df['Time_Index'] = np.arange(len(df))  # Convert time into numerical index

# Train Linear Regression model
X = df[['Time_Index']]
y = df['Used Memory']
model = LinearRegression()
model.fit(X, y)

# Predict future memory usage
future_time_indices = np.arange(len(df), len(df) + 50).reshape(-1, 1)  # Predict next 50 time points
future_predictions = model.predict(pd.DataFrame(future_time_indices, columns=['Time_Index']))

# Create future timestamps
future_timestamps = pd.date_range(start=df['Timestamp'].iloc[-1], periods=51, freq='min')[1:]  # Exclude last known timestamp
future_df = pd.DataFrame({'Timestamp': future_timestamps, 'Predicted Memory': future_predictions})

# Initialize Dash app
app = dash.Dash(__name__)

# Layout of the dashboard
app.layout = html.Div(children=[
    html.H1("Memory Consumption Forecasting Dashboard", style={'text-align': 'center'}),

    # Auto-refresh every 30 seconds
    dcc.Interval(id='interval-component', interval=30 * 1000, n_intervals=0),

    # Memory Usage Over Time

    # Memory Forecast Graph
    dcc.Graph(id='memory_forecast_graph'),

    # Bottleneck Identification
    html.H3("Bottleneck Analysis"),
    dcc.Graph(id='bottleneck_graph'),

    # Additional Graphs
    html.H3("Additional Insights"),

    # Memory vs Free Memory (Scatter Plot)
    dcc.Graph(id='memory_vs_free_memory'),

    # Histogram of Memory Usage
    dcc.Graph(id='memory_histogram'),

    # Box Plot of Memory Usage by Hour
    dcc.Graph(id='memory_box_plot'),

    # Correlation Heatmap
    dcc.Graph(id='correlation_heatmap'),

    # Real-Time Memory Usage Bar Chart
    dcc.Graph(id='real_time_memory_usage')
])


# Callback to update the memory usage graph


# Callback to update the memory forecast graph
@app.callback(
    Output('memory_forecast_graph', 'figure'),
    [Input('interval-component', 'n_intervals')]
)
def update_forecast_graph(n):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Timestamp'], y=df['Used Memory'], mode='lines', name="Actual Memory Usage"))
    fig.add_trace(go.Scatter(
        x=future_df['Timestamp'],
        y=future_df['Predicted Memory'],
        mode='lines+markers',
        name="Predicted Memory Usage",
        line=dict(dash='dot', width=3),
        marker=dict(size=6, color='red')
    ))
    fig.update_layout(title="Memory Usage Forecast (Linear Regression)")
    return fig


# Callback to generate bottleneck analysis
@app.callback(
    Output('bottleneck_graph', 'figure'),
    [Input('interval-component', 'n_intervals')]
)
def update_bottleneck_graph(n):
    # Create a string column for easy plotting
    df['High Memory Usage'] = np.where(df['Used Memory'] > df['Used Memory'].quantile(0.90), "High", "Normal")

    fig = px.scatter(
        df,
        x="Timestamp",
        y="Used Memory",
        color="High Memory Usage",
        title="Bottleneck Identification: High Memory Usage Instances",
        labels={'High Memory Usage': 'Bottleneck Detected'},
        opacity=0.7,
        color_discrete_map={"High": "red", "Normal": "blue"}
    )

    return fig


# Callback to generate Memory vs Free Memory graph
@app.callback(
    Output('memory_vs_free_memory', 'figure'),
    [Input('interval-component', 'n_intervals')]
)
def update_memory_vs_free_memory(n):
    # Assuming you have a "Free Memory" column in the dataset
    fig = px.scatter(df, x='Used Memory', y='Free Memory', color='Disk Utilization (%)', title="Used Memory vs Free Memory")
    return fig


# Callback to generate Histogram of Memory Usage
@app.callback(
    Output('memory_histogram', 'figure'),
    [Input('interval-component', 'n_intervals')]
)
def update_memory_histogram(n):
    fig = px.histogram(df, x='Used Memory', nbins=30, title="Memory Usage Distribution")
    return fig


# Callback to generate Box Plot of Memory Usage by Hour
@app.callback(
    Output('memory_box_plot', 'figure'),
    [Input('interval-component', 'n_intervals')]
)
def update_memory_box_plot(n):
    # Extract the hour from the Timestamp
    df['Hour'] = df['Timestamp'].dt.hour

    fig = px.box(df, x='Hour', y='Used Memory', title="Memory Usage by Hour")
    return fig


# Callback to generate Correlation Heatmap
@app.callback(
    Output('correlation_heatmap', 'figure'),
    [Input('interval-component', 'n_intervals')]
)
def update_correlation_heatmap(n):
    df_numeric = df.select_dtypes(include=['number'])
    corr_matrix = df_numeric.corr()

    # Use Plotly for heatmap visualization
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='Viridis'
    ))
    fig.update_layout(title="Correlation Heatmap")
    return fig


# Callback to generate Real-Time Memory Usage Bar Chart
@app.callback(
    Output('real_time_memory_usage', 'figure'),
    [Input('interval-component', 'n_intervals')]
)
def update_real_time_memory_usage(n):
    # Get the latest data point
    latest_data = df.iloc[-10:]  # Show last 10 data points
    fig = px.bar(latest_data, x='Timestamp', y='Used Memory', title="Real-Time Memory Usage")
    return fig


# Run the app
if __name__ == '__main__':
    app.run_server(debug=True, port=8051)
