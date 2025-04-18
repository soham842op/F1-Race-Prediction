import os
import pandas as pd
import fastf1
import numpy as np 

# Cache folder 
cache_folder = 'cache_folder'
if not os.path.exists(cache_folder):
    os.makedirs(cache_folder)

fastf1.Cache.enable_cache(cache_folder)

# Get data from China GP 2025
session = fastf1.get_session(2025, 'China', 'R')
session.load()

print('Cache enabled and race data loaded successfully')

# Deciding the data I want to load 
laps = session.laps
df = laps[['Driver', 'Team', 'LapNumber', 'LapTime', 'Compound', 'TyreLife', 
           'Position', 'Sector1Time', 'Sector2Time', 'Sector3Time', 
           'SpeedI1', 'SpeedI2', 'SpeedFL', 'SpeedST',
           'IsPersonalBest', 'TrackStatus', 'FreshTyre']]

# Convert lap times to seconds for easier analysis
df['LapTimeSeconds'] = df['LapTime'].dt.total_seconds()

# Safely calculate sector times in seconds
df['Sector1Seconds'] = df['Sector1Time'].apply(lambda x: x.total_seconds() if pd.notna(x) else np.nan)
df['Sector2Seconds'] = df['Sector2Time'].apply(lambda x: x.total_seconds() if pd.notna(x) else np.nan)
df['Sector3Seconds'] = df['Sector3Time'].apply(lambda x: x.total_seconds() if pd.notna(x) else np.nan)

# Create aggregate metrics
df['TotalSectorTime'] = df['Sector1Seconds'] + df['Sector2Seconds'] + df['Sector3Seconds']

# Track performance consistency
df['Consistency'] = df.groupby('Driver')['LapTimeSeconds'].transform(lambda x: np.std(x))
df['DeltaToPrevLap'] = df.groupby('Driver')['LapTimeSeconds'].diff()

# Try to integrate weather data
if hasattr(session, 'weather_data'):
    weather = session.weather_data
    # Find common time column
    time_columns = [col for col in laps.columns if 'Time' in col or 'time' in col.lower()]
    if time_columns and 'Time' in weather.columns:
        merge_col = time_columns[0]
        laps_copy = laps.copy()
        laps_copy['MergeTime'] = laps_copy[merge_col]
        weather_subset = weather[['Time', 'AirTemp', 'TrackTemp', 'Humidity', 'WindSpeed']].copy()
        weather_subset['MergeTime'] = weather_subset['Time']
        
        # Merge weather data
        try:
            df = pd.merge_asof(
                laps_copy.sort_values('MergeTime'),
                weather_subset.sort_values('MergeTime'),
                on='MergeTime',
                direction='nearest'
            )
            print("Weather data successfully merged")
        except Exception as e:
            print(f"Error merging weather data: {e}")
    else:
        print("Required time columns not found for weather merge")
else:
    print("Weather data not available for this session")

# Get qualifying data for starting positions
try:
    quali = fastf1.get_session(2025, 'China', 'Q')
    quali.load()
    quali_results = quali.results[['Abbreviation', 'Position']]
    quali_results.rename(columns={'Abbreviation': 'Driver', 'Position': 'GridPosition'}, inplace=True)
    df = pd.merge(df, quali_results, on='Driver', how='left')
    print("Qualifying data successfully merged")
except Exception as e:
    print(f"Could not load qualifying data: {e}")

# Function to get car telemetry for a specific lap
def get_telemetry_for_lap(driver, lap_number):
    try:
        lap = laps.pick_driver(driver).pick_lap(lap_number)
        telemetry = lap.get_telemetry()
        return telemetry
    except Exception as e:
        print(f"Error getting telemetry for {driver}, lap {lap_number}: {e}")
        return None

# Function to analyze fastest laps
def analyze_fastest_laps():
    fastest_laps = laps.pick_fastest_per_driver()
    telemetry_data = {}
    
    for _, row in fastest_laps.iterrows():
        driver = row['Driver']
        lap_n = row['LapNumber']
        print(f"Getting telemetry for {driver}'s fastest lap ({lap_n})")
        telemetry = get_telemetry_for_lap(driver, lap_n)
        if telemetry is not None:
            # Store basic telemetry statistics
            telemetry_data[driver] = {
                'max_speed': telemetry['Speed'].max(),
                'avg_speed': telemetry['Speed'].mean(),
                'max_rpm': telemetry['RPM'].max() if 'RPM' in telemetry.columns else None,
                'braking_zones': (telemetry['Brake'].diff() > 0).sum() if 'Brake' in telemetry.columns else None
            }
    
    return pd.DataFrame.from_dict(telemetry_data, orient='index')

# Function to compare drivers
def compare_drivers(df, driver1, driver2):
    driver1_data = df[df['Driver'] == driver1]
    driver2_data = df[df['Driver'] == driver2]
    
    if driver1_data.empty or driver2_data.empty:
        return f"One or both drivers ({driver1}, {driver2}) not found in the data"
    
    comparison = pd.DataFrame({
        'Metric': ['Avg Lap Time', 'Best Lap Time', 'Avg Speed', 'Consistency'],
        f'{driver1}': [
            driver1_data['LapTimeSeconds'].mean(),
            driver1_data['LapTimeSeconds'].min(),
            driver1_data[['SpeedI1', 'SpeedI2', 'SpeedFL', 'SpeedST']].mean().mean(),
            driver1_data['Consistency'].iloc[0] if not driver1_data.empty else None
        ],
        f'{driver2}': [
            driver2_data['LapTimeSeconds'].mean(),
            driver2_data['LapTimeSeconds'].min(),
            driver2_data[['SpeedI1', 'SpeedI2', 'SpeedFL', 'SpeedST']].mean().mean(),
            driver2_data['Consistency'].iloc[0] if not driver2_data.empty else None
        ]
    })
    
    return comparison

# Function to analyze tire performance
def analyze_tire_performance(df):
    # Group by compound and calculate statistics
    tire_stats = df.groupby('Compound').agg({
        'LapTimeSeconds': ['mean', 'min', 'std'],
        'TyreLife': ['mean', 'max']
    })
    
    # Calculate degradation (estimated as time increase per lap of tire life)
    deg_data = []
    for compound in df['Compound'].unique():
        if pd.isna(compound):
            continue
        
        compound_data = df[df['Compound'] == compound].copy()
        if len(compound_data) > 10:  # Need enough data points
            compound_data = compound_data.sort_values('TyreLife')
            # Simple linear regression to find degradation rate
            from scipy import stats
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                compound_data['TyreLife'], 
                compound_data['LapTimeSeconds']
            )
            deg_data.append({
                'Compound': compound, 
                'Degradation_per_lap': slope,
                'R_squared': r_value**2
            })
    
    degradation_df = pd.DataFrame(deg_data)
    
    return tire_stats, degradation_df

# Main analysis
print("\nBasic statistics:")
print(df.describe())

# Uncomment to run additional analyses
# fastest_telemetry = analyze_fastest_laps()
# print("\nFastest lap telemetry analysis:")
# print(fastest_telemetry)

# Example driver comparison (adjust driver codes as needed)
# If you know specific drivers to compare, uncomment and modify:
# print("\nDriver comparison:")
# print(compare_drivers(df, 'VER', 'HAM'))  # Adjust driver codes as needed

# tire_stats, tire_degradation = analyze_tire_performance(df)
# print("\nTire compound performance:")
# print(tire_stats)
# print("\nTire degradation rates:")
# print(tire_degradation)

print("\nFull dataset sample:")
print(df.head())

