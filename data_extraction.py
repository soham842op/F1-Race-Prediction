import os
import pandas as pd
import fastf1
import numpy as np 

#cache folder 
cache_folder = 'cache_folder'
if not os.path.exists(cache_folder):
    os.makedirs(cache_folder)

fastf1.Cache.enable_cache(cache_folder)



#Get data from China GP 2025
session= fastf1.get_session(2025,'China','R')
session.load()

print('Cache enabled and race data loaded successfully')

#Deciding the data I want to load 
laps=session.laps
df = laps[['Driver', 'Team', 'LapNumber', 'LapTime', 'Compound', 'TyreLife', 
           'Position', 'Sector1Time', 'Sector2Time', 'Sector3Time', 
           'SpeedI1', 'SpeedI2', 'SpeedFL', 'SpeedST',
           'IsPersonalBest', 'TrackStatus', 'FreshTyre']]



print(df)


