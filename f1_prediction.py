import fastf1
import pandas as pd
import numpy as np
import json
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from tabulate import tabulate 

# Enable FastF1 Cache
fastf1.Cache.enable_cache("cache")

# Path to JSON file containing real qualifying times
QUALI_DATA_FILE = "quali_times_2025.json"

# Global constants for drivers and qualifying times
DEFAULT_DRIVERS = [
    "Lando Norris", "Oscar Piastri", "Max Verstappen", "George Russell",
    "Yuki Tsunoda", "Alexander Albon", "Charles Leclerc", "Lewis Hamilton",
    "Pierre Gasly", "Carlos Sainz", "Fernando Alonso", "Lance Stroll"
]

DEFAULT_QUALIFYING_TIMES = [
    88.5, 88.7, 89.0, 89.2,
    89.5, 89.7, 90.0, 90.2,
    90.5, 90.7, 91.0, 91.2
]

# Standard driver mapping
DRIVER_MAPPING = {
    "Lando Norris": "NOR",
    "Oscar Piastri": "PIA",
    "Max Verstappen": "VER",
    "George Russell": "RUS",
    "Yuki Tsunoda": "TSU",
    "Alexander Albon": "ALB",
    "Charles Leclerc": "LEC",
    "Lewis Hamilton": "HAM",
    "Pierre Gasly": "GAS",
    "Carlos Sainz": "SAI",
    "Lance Stroll": "STR",
    "Fernando Alonso": "ALO",
    "Sergio Perez": "PER",
    "Kevin Magnussen": "MAG",
    "Nico Hulkenberg": "HUL",
    "Valtteri Bottas": "BOT",
    "Zhou Guanyu": "ZHO",
    "Esteban Ocon": "OCO",
    "Daniel Ricciardo": "RIC",
    "Logan Sargeant": "SAR",
}

# Dictionary to store our trained models (one per circuit)
models = {}

def load_qualifying_data():
    """
    Load qualifying data from JSON file.
    If file doesn't exist, create an empty one.
    
    Returns:
    - dict: Qualifying data keyed by GP name
    """
    if not os.path.exists(QUALI_DATA_FILE):
        # Create empty file if it doesn't exist
        with open(QUALI_DATA_FILE, 'w') as f:
            json.dump({}, f, indent=2)
        return {}
    
    try:
        with open(QUALI_DATA_FILE, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading qualifying data: {e}")
        return {}

def save_qualifying_data(quali_data):
    """
    Save qualifying data to JSON file.
    
    Parameters:
    - quali_data: Dict of qualifying data to save
    """
    try:
        with open(QUALI_DATA_FILE, 'w') as f:
            json.dump(quali_data, f, indent=2)
        return True
    except Exception as e:
        print(f"Error saving qualifying data: {e}")
        return False

def load_race_and_weather(year, gp_name):
    """
    Loads the Race session for a specific GP in 'year' using FastF1.
    Returns a tuple:
      - laps: a DataFrame of laps (with Driver, LapTime, Sector1Time, Sector2Time, Sector3Time)
      - agg_weather: a dictionary of aggregated weather features for the session
    """
    try:
        session = fastf1.get_session(year, gp_name, 'R')
        session.load()
        
        # Extract laps and clean up: drop rows with missing lap/sector times
        laps = session.laps[['Driver', 'LapTime', 'Sector1Time', 'Sector2Time', 'Sector3Time']].copy()
        laps.dropna(subset=['LapTime','Sector1Time','Sector2Time','Sector3Time'], inplace=True)
        laps['Year'] = year
        
        # Get weather data; drop rows with missing weather info if any
        weather = session.weather_data.copy().dropna()
        
        # Compute min and max weather features
        vars_to_agg = ['AirTemp', 'Humidity', 'Pressure', 'TrackTemp', 'WindSpeed']
        agg_weather = {}
        for var in vars_to_agg:
            if var in weather.columns:
                agg_weather[f"{var}_min"] = weather[var].min()
                agg_weather[f"{var}_max"] = weather[var].max()
            else:
                # If weather data is missing, use zero or some fallback
                agg_weather[f"{var}_min"] = 0
                agg_weather[f"{var}_max"] = 0
        
        # Add these aggregated weather features as new columns to each lap record.
        for key, value in agg_weather.items():
            laps[key] = value

        return laps, agg_weather
    except Exception as e:
        print(f"Error loading data for {gp_name} {year}: {str(e)}")
        return None, None

def train_model_for_gp(gp_name):
    """
    Trains an ML model for a specific Grand Prix using historical data.
    Using min and max weather values for better prediction accuracy.
    
    Returns:
        model               - The trained GradientBoostingRegressor model
        avg_sector_times    - Average sector times per driver
        weather_features    - Dictionary of aggregated weather features
        feature_cols        - The list of feature column names used
    """
    # Try to load data from the last two completed seasons
    laps_previous_years = []
    for year in [ 2023, 2024]:
        laps, _ = load_race_and_weather(year, gp_name)
        if laps is not None and not laps.empty:
            laps_previous_years.append(laps)
    
    if not laps_previous_years:
        return None, None, None, None
    
    # Combine data from all years
    laps_all = pd.concat(laps_previous_years, ignore_index=True)
    
    # Additional cleaning: drop any rows with missing values
    laps_all.dropna(inplace=True)
    
    # Convert timedeltas to seconds
    for col in ["LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]:
        laps_all[f"{col} (s)"] = laps_all[col].dt.total_seconds()
    
    # Compute average sector times per driver
    avg_sector_times = (
        laps_all
        .groupby('Driver')[["Sector1Time (s)", "Sector2Time (s)", "Sector3Time (s)"]]
        .mean()
        .reset_index()
    )
    
    # Extract min and max weather features from the first row
    weather_features = {}
    for var in ['AirTemp', 'Humidity', 'Pressure', 'TrackTemp', 'WindSpeed']:
        for stat in ['min', 'max']:
            key = f"{var}_{stat}"
            if key in laps_all.columns:
                weather_features[key] = laps_all[key].iloc[0]
    
    # Define target: average race lap time per driver
    avg_race_times = laps_all.groupby('Driver')["LapTime (s)"].mean()
    
    # Prepare fixed qualifying times using global constants
    qual_2025 = pd.DataFrame({
        "Driver": DEFAULT_DRIVERS,
        "QualifyingTime (s)": DEFAULT_QUALIFYING_TIMES
    })
    # Add driver codes
    qual_2025["DriverCode"] = qual_2025["Driver"].map(DRIVER_MAPPING)
    
    # Merge with average sector times
    merged_data = pd.merge(
        qual_2025,
        avg_sector_times,
        left_on="DriverCode",
        right_on="Driver",
        how="inner"
    )
    
    # Add weather features to the merged data
    for key, value in weather_features.items():
        merged_data[key] = value
    
    # Define feature columns
    feature_cols = [
        "QualifyingTime (s)",
        "Sector1Time (s)", "Sector2Time (s)", "Sector3Time (s)",
    ]
    
    # Add weather feature columns
    for var in ['AirTemp', 'Humidity', 'Pressure', 'TrackTemp', 'WindSpeed']:
        for stat in ['min', 'max']:
            key = f"{var}_{stat}"
            if key in weather_features:
                feature_cols.append(key)
    
    X = merged_data[feature_cols].copy()
    
    # Merge with target variable
    merged_data = pd.merge(
        merged_data,
        avg_race_times,
        left_on="DriverCode",
        right_index=True,
        how="left"
    )
    y = merged_data["LapTime (s)"]
    
    # Check for missing target values
    if y.isna().any():
        print(f"Warning: Some drivers not matched for avg race-lap time for {gp_name}. Check driver codes.")
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train model
    model = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred_test = model.predict(X_test)
    mae_test = mean_absolute_error(y_test, y_pred_test)
    r2 = r2_score(y_test, y_pred_test)
    print(f"Model for {gp_name}: Test MAE: {mae_test:.2f} seconds, R² Score: {r2:.2f}")
    
    return model, avg_sector_times, weather_features, feature_cols

def predict_race_results(gp_name, drivers=None):
    """
    Predicts race results for a given Grand Prix (gp_name).
    If available, uses real qualifying times from JSON for the drivers.
    Otherwise uses default drivers/times.
    """
    # Check if we already have a trained model for this GP
    if gp_name not in models:
        model, avg_sector_times, weather_features, feature_cols = train_model_for_gp(gp_name)
        
        if model is None:
            return {"error": f"Could not train model for {gp_name}. Historical data might not be available."}
        
        # Cache the trained model
        models[gp_name] = {
            "model": model,
            "avg_sector_times": avg_sector_times,
            "weather_features": weather_features,
            "feature_cols": feature_cols
        }
    else:
        model_data = models[gp_name]
        model = model_data["model"]
        avg_sector_times = model_data["avg_sector_times"]
        weather_features = model_data["weather_features"]
        feature_cols = model_data["feature_cols"]
    
    # Load possible real qualifying data
    quali_data = load_qualifying_data()
    real_quali_times = quali_data.get(gp_name, {})
    has_real_data = bool(real_quali_times)
    
    # Decide which drivers to use
    if drivers is None:
        if has_real_data:
            # If we have real data, use those drivers
            drivers = list(real_quali_times.keys())
        else:
            # Otherwise use default drivers
            drivers = DEFAULT_DRIVERS
    
    # Build list of qualifying times
    if has_real_data:
        # Use real qualifying times for the available drivers
        drivers_with_times = []
        qual_times = []
        
        for driver in drivers:
            if driver in real_quali_times:
                drivers_with_times.append(driver)
                qual_times.append(real_quali_times[driver])
        
        # Update drivers to the ones we have real times for
        drivers = drivers_with_times
    else:
        # Use default times (extend if needed)
        qual_times = DEFAULT_QUALIFYING_TIMES[:len(drivers)]
        if len(drivers) > len(qual_times):
            last_time = qual_times[-1] if qual_times else 88.5
            increment = 0.2
            additional_times = [last_time + increment * (i+1) for i in range(len(drivers) - len(qual_times))]
            qual_times.extend(additional_times)
    
    # Create qualifying DataFrame
    qual_df = pd.DataFrame({
        "Driver": drivers,
        "QualifyingTime (s)": qual_times
    })
    
    # Map driver names to codes
    qual_df["DriverCode"] = qual_df["Driver"].map(lambda x: DRIVER_MAPPING.get(x, "UNK"))
    
    # Merge with average sector times
    merged_data = pd.merge(
        qual_df,
        avg_sector_times,
        left_on="DriverCode",
        right_on="Driver",
        how="inner"
    )
    
    # Add weather features
    for key, value in weather_features.items():
        merged_data[key] = value
    
    # Prepare final features
    X_pred = merged_data[feature_cols].copy()
    
    # Predict race times
    pred_times = model.predict(X_pred)
    merged_data["PredictedRaceTime (s)"] = pred_times
    
    # Sort by predicted time ascending (fastest = 1st)
    merged_data = merged_data.sort_values("PredictedRaceTime (s)", ascending=True).reset_index(drop=True)
    
    # Build results
    results = []
    fastest_time = merged_data["PredictedRaceTime (s)"].iloc[0]
    for i, row in merged_data.iterrows():
        results.append({
            "Position": i + 1,
            "Driver": row["Driver_x"] if "Driver_x" in row else row["Driver"],
            "QualifyingTime (s)": float(row["QualifyingTime (s)"]),
            "PredictedRaceTime (s)": float(row["PredictedRaceTime (s)"]),
            "Gap to P1 (s)": float(row["PredictedRaceTime (s)"] - fastest_time)
        })
    
    return results

if __name__ == "__main__":
    # Example usage of the script
    gp_name = "Japanese"  # Change this to whatever GP you want to predict
    
    # By default, uses real quali times if they exist, else default drivers
    results = predict_race_results(gp_name)
    
    if isinstance(results, dict) and "error" in results:
        print(results["error"])
    else:
        # Print as a table using tabulate
        print(f"\nPredicted Race Results for {gp_name}:\n")
        print(tabulate(results, headers="keys", tablefmt="fancy_grid"))
