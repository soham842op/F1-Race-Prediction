# F1 Race Prediction Analysis

A data science project to predict Formula 1 race outcomes using FastF1 API and machine learning models.

## Features
- Data extraction from FastF1 API
- Analysis of lap times, tire performance, and weather conditions
- Predictive modeling for race outcomes
- Visualization dashboard using Streamlit



## Day 1: Data Extraction & Exploration

### Accomplished Today:
- Set up the FastF1 API environment with local cache
- Successfully extracted detailed F1 race data from the China GP
- Identified key performance metrics for analysis:
  - Lap times and position tracking
  - Sector breakdowns (Sector1, Sector2, Sector3)
  - Tire compound information and tire life tracking
  - Speed trap data (SpeedI1, SpeedI2, SpeedFL, SpeedST)
  - Track conditions and driver performance indicators


### Day 2: Advanced Data Processing & Analysis Framework
### Accomplished Today:

- Enhanced the data extraction process with derived performance metrics
- Added conversion of lap and sector times to seconds for easier numerical analysis
- Implemented performance consistency tracking across drivers and laps
- Developed functions for in-depth race analysis:

- get_telemetry_for_lap(): Extracts detailed car telemetry data
- analyze_fastest_laps(): Compares peak performance across drivers
- compare_drivers(): Direct head-to-head driver performance comparison
- analyze_tire_performance(): Evaluates compound effectiveness and degradation rates
