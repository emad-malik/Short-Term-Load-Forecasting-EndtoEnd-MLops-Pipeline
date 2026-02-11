import pandas as pd
import numpy as np
import os
import json
import glob
import zipfile
import warnings
warnings.filterwarnings('ignore')

RAW_DATA_PATH = "data/raw/"
PROCESSED_DATA_PATH = "data/processed/"
ARCHIVE_PATH = "data/raw/data.zip"

def extract_data(zip_path, extract_to):
    """Unzips the raw data if not already extracted."""
    if not os.path.exists(extract_to):
        os.makedirs(extract_to)
    
    if os.path.exists(zip_path):
        print(f"Extracting {zip_path}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
    else:
        print(f"Warning: Zip file not found at {zip_path}. Assuming data is already extracted.")

def drop_columns_missing_threshold(df, threshold_percent):
    """Drops columns with missing values above a certain percentage."""
    missing_fraction = df.isna().mean()
    cols_to_keep = missing_fraction[missing_fraction <= threshold_percent / 100].index
    return df[cols_to_keep]

def get_season(month):
    """Returns season based on month index."""
    if month in [12, 1, 2]:
        return 'winter'
    elif month in [3, 4, 5]:
        return 'spring'
    elif month in [6, 7, 8]:
        return 'summer'
    else:
        return 'fall'

def process_energy_data(data_dir):
    """Reads, merges, and cleans EIA930 energy data."""
    print("Processing Energy Data...")
    
    # 1. Load and Concatenate Subregion Data
    subregion_files = glob.glob(os.path.join(data_dir, 'EIA930_SUBREGION_*.csv'))
    subregion_list = []
    for file in subregion_files:
        df = pd.read_csv(file, parse_dates=['Local Time at End of Hour', 'UTC Time at End of Hour'])
        subregion_list.append(df)
    eia_subregion_all = pd.concat(subregion_list, ignore_index=True)

    # 2. Load and Concatenate Balance Data
    balance_files = glob.glob(os.path.join(data_dir, 'EIA930_BALANCE_*.csv'))
    balance_list = []
    for file in balance_files:
        df = pd.read_csv(file, parse_dates=['Local Time at End of Hour', 'UTC Time at End of Hour'])
        balance_list.append(df)
    eia_balance_all = pd.concat(balance_list, ignore_index=True)

    # 3. Drop columns with high missing values
    # Subregion has 'Demand (MW)' with 9.62% missing, so use 10% threshold
    subregion_clean = drop_columns_missing_threshold(eia_subregion_all, 10)
    balance_clean = drop_columns_missing_threshold(eia_balance_all, 20)

    # 4. Merge Subregion and Balance Data
    merged_df = pd.merge(
        subregion_clean,
        balance_clean,
        on=['Balancing Authority', 'Data Date', 'Hour Number'],
        how='inner',
        suffixes=('_subregion', '_balance')
    )

    # 5. Clean Numeric Columns (Remove commas, handle NaNs)
    numeric_cols_energy = [
        'Demand Forecast (MW)', 'Demand (MW)', 'Net Generation (MW)',
        'Total Interchange (MW)', 'Sum(Valid DIBAs) (MW)',
        'Demand (MW) (Adjusted)', 'Net Generation (MW) (Adjusted)'
    ]

    for col in numeric_cols_energy:
        # Check if column exists before processing
        if col in merged_df.columns:
            if merged_df[col].dtype == 'object':
                merged_df[col] = merged_df[col].str.replace(',', '', regex=False)
            merged_df[col] = pd.to_numeric(merged_df[col], errors='coerce')

    # Drop rows where critical targets are missing
    merged_df.dropna(subset=['Demand Forecast (MW)', 'Demand (MW) (Adjusted)'], inplace=True)

    # Fill remaining NaNs with mean grouped by Sub-Region and Hour
    for col in numeric_cols_energy:
        if col in merged_df.columns:
            merged_df[col] = merged_df.groupby(['Sub-Region', 'Hour Number'])[col].transform(
                lambda x: x.fillna(x.mean())
            )

    # 6. Feature Engineering (Time)
    # Use the balance timestamp as the primary time
    time_col = 'Local Time at End of Hour_balance'
    merged_df['hour'] = merged_df[time_col].dt.hour
    merged_df['day_of_week'] = merged_df[time_col].dt.dayofweek
    merged_df['month'] = merged_df[time_col].dt.month
    merged_df['season'] = merged_df['month'].apply(get_season)

    return merged_df

def process_weather_data(data_dir):
    """Reads, merges, and cleans JSON weather data."""
    print("Processing Weather Data...")
    
    city_files = [
        'san_diego.json', 'la.json', 'philadelphia.json', 'san_antonio.json',
        'dallas.json', 'phoenix.json', 'nyc.json', 'san_jose.json',
        'seattle.json', 'houston.json'
    ]

    merged_weather = []

    for file in city_files:
        file_path = os.path.join(data_dir, file)
        if os.path.exists(file_path):
            df = pd.read_json(file_path)
            # Extract city name from filename
            city_name = file.replace('.json', '')
            df['city'] = city_name
            # Convert 'time' (unix timestamp) to datetime
            df['local_time'] = pd.to_datetime(df['time'], unit='s')
            merged_weather.append(df)
        else:
            print(f"Warning: {file} not found.")

    if not merged_weather:
        return pd.DataFrame() # Return empty if no files found

    weather_all = pd.concat(merged_weather, ignore_index=True)

    # 1. Drop High Missing Value Columns
    weather_clean = drop_columns_missing_threshold(weather_all, 10)

    # 2. Clean Specific Columns
    # Drop rows without summary or icon
    weather_clean.dropna(subset=['summary', 'icon'], inplace=True)

    # Fill numeric NaNs with mean
    numeric_cols_weather = [
        'precipIntensity', 'precipProbability', 'temperature', 'apparentTemperature',
        'humidity', 'pressure', 'windSpeed', 'windGust', 'windBearing',
        'cloudCover', 'uvIndex', 'visibility'
    ]
    
    for col in numeric_cols_weather:
        if col in weather_clean.columns:
            if weather_clean[col].isnull().any():
                weather_clean[col].fillna(weather_clean[col].mean(), inplace=True)

    # 3. Feature Engineering
    weather_clean['hour'] = weather_clean['local_time'].dt.hour
    weather_clean['day_of_week'] = weather_clean['local_time'].dt.dayofweek
    weather_clean['month'] = weather_clean['local_time'].dt.month
    weather_clean['season'] = weather_clean['month'].apply(get_season)

    return weather_clean

def run_etl():
    # Ensure directories exist
    if not os.path.exists(PROCESSED_DATA_PATH):
        os.makedirs(PROCESSED_DATA_PATH)

    # 1. Extract Data
    extract_data(ARCHIVE_PATH, RAW_DATA_PATH)

    # 2. Process Energy Data
    energy_df = process_energy_data(RAW_DATA_PATH)
    energy_output_path = os.path.join(PROCESSED_DATA_PATH, 'merged_energy_data.csv')
    energy_df.to_csv(energy_output_path, index=False)
    print(f"Saved cleaned energy data to {energy_output_path} (Shape: {energy_df.shape})")

    # 3. Process Weather Data
    weather_df = process_weather_data(RAW_DATA_PATH)
    weather_output_path = os.path.join(PROCESSED_DATA_PATH, 'weather_merged.csv')
    weather_df.to_csv(weather_output_path, index=False)
    print(f"Saved cleaned weather data to {weather_output_path} (Shape: {weather_df.shape})")

if __name__ == "__main__":
    run_etl()