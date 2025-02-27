import requests
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import datetime


# --- Fetch historical weather data from Open-Meteo API ---
print("Fetching historical weather data...")
url = "https://archive-api.open-meteo.com/v1/archive"
params = {
    "latitude": 52.52,
    "longitude": 13.41,
    "start_date": "2013-01-01",
    "end_date": "2023-12-31",
    "hourly": ["temperature_2m", "precipitation"],
    "timezone": "Europe/Berlin"
}
response = requests.get(url, params=params)
data = response.json()

# Convert JSON to DataFrame
df = pd.DataFrame({
    "datetime": pd.to_datetime(data["hourly"]["time"]),
    "temperature": data["hourly"]["temperature_2m"],
    "precipitation": data["hourly"]["precipitation"]
})


# --- Train and Evaluate Machine Learning Models ---
print("Training and evaluating models...")
# Set datetime as index
df.set_index("datetime", inplace=True)

# Feature Engineering
df["hour"] = df.index.hour
df["day_of_year"] = df.index.dayofyear
df["month"] = df.index.month

# Define features and target
X = df[["hour", "day_of_year", "month", "precipitation"]]
y = df["temperature"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define models
models = {
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
    "Linear Regression": LinearRegression(),
}

# Train and evaluate models
results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    results[name] = mae
    print(f"{name} - Mean Absolute Error: {mae:.2f}°C")

# Select best model
best_model_name = min(results, key=results.get)
best_model = models[best_model_name]
print(f"Best Model: {best_model_name}")

# --- Make Predictions ---
print("Making predictions...")
# Predict for a future date
future_date = datetime.datetime(2025, 2, 21)  # Example future date
future_features = pd.DataFrame({
    "hour": [12],  # Noon
    "day_of_year": [future_date.timetuple().tm_yday],
    "month": [future_date.month],
    "precipitation": [0]  # Assuming no precipitation
})
future_prediction = best_model.predict(future_features)
print(f"Predicted Temperature for {future_date.strftime('%Y-%m-%d')} using {best_model_name}: {future_prediction[0]:.2f}°C")

# --- Retrieve Actual Temperature for Comparison ---
print("Retrieving actual temperature for comparison...")
# Retrieve actual temperature for comparison
actual_url = "https://api.open-meteo.com/v1/forecast"
actual_params = {
    "latitude": 52.52,
    "longitude": 13.41,
    "hourly": ["temperature_2m"],
    "timezone": "Europe/Berlin",
    "start_date": future_date.strftime("%Y-%m-%d"),
    "end_date": future_date.strftime("%Y-%m-%d")
}
actual_response = requests.get(actual_url, params=actual_params)
actual_data = actual_response.json()

# Check if actual temperature data is available
if "hourly" in actual_data and "temperature_2m" in actual_data["hourly"]:
    actual_temperature = actual_data["hourly"]["temperature_2m"][12]  # Noon temperature
    print(f"Actual Temperature for {future_date.strftime('%Y-%m-%d')}: {actual_temperature:.2f}°C")
    print(f"Prediction Error: {abs(actual_temperature - future_prediction[0]):.2f}°C")
else:
    print("Actual temperature data not available for comparison.")
