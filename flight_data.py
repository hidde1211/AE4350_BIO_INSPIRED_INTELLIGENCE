import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import geopy
from scipy.interpolate import interp1d
from wind_sim import interpolated_wind_data,get_headwind

param_dict = pd.read_excel("aircraft_params.xlsx")
name,M,FL0,FL_max,m0,mf0,mzf,CD0,k,S,TSFC = param_dict.values[0]

def calculate_initial_compass_bearing(lat1, lon1, lat2, lon2):
    # Convert latitude and longitude from degrees to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    # Calculate the differences in longitude and latitude
    d_lon = lon2 - lon1

    # Calculate the initial bearing
    y = np.sin(d_lon) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(d_lon)
    initial_bearing = np.arctan2(y, x)

    # Convert the bearing from radians to degrees
    initial_bearing = np.degrees(initial_bearing)

    # Normalize the initial bearing to the range [0, 360)
    initial_bearing = (initial_bearing + 360) % 360

    return initial_bearing


def get_flight_path_data(desired_spacing_km):
    #load dataframe only keep lat,lon,heading,alt
    flight_path = pd.read_excel("BA115_17_04_2023.xlsx").reset_index(drop=True).dropna(how="all")[["lat","lon","alt_ft"]]

    #only keep cruise flight
    flight_path = flight_path.loc[43:542]
    flight_path.index = np.arange(len(flight_path))

    #add missing values of altitude by forward fill
    flight_path["alt_ft"] = flight_path["alt_ft"].fillna(method="bfill")

    #determine distances between coordinates
    for index in range(1,len(flight_path)):
        old_lat,old_lon = flight_path.loc[index-1,["lat","lon"]]
        lat,lon = flight_path.loc[index,["lat","lon"]]
        dx = geopy.distance.geodesic((old_lat,old_lon),(lat,lon)).km
        flight_path.loc[index,"x"] = dx
    flight_path["x"] = flight_path["x"].fillna(0.).cumsum()

    #create interpolation functions for distance
    interp_latitude = interp1d(flight_path["x"].values, flight_path["lat"].values, kind='linear')
    interp_longitude = interp1d(flight_path["x"].values, flight_path["lon"].values, kind='linear')
    interp_altitude = interp1d(flight_path["x"].values, flight_path["alt_ft"].values, kind='linear')

    # Define the desired even spacing in kilometers
    evenly_spaced_distances = np.arange(0, flight_path["x"].values[-1], desired_spacing_km)

    # Interpolate new latitude, longitude and altitude coordinates
    evenly_spaced_latitudes = interp_latitude(evenly_spaced_distances)
    evenly_spaced_longitudes = interp_longitude(evenly_spaced_distances)
    evenly_spaced_altitudes = interp_altitude(evenly_spaced_distances)
    df = pd.DataFrame([evenly_spaced_latitudes,evenly_spaced_longitudes,evenly_spaced_altitudes],index = ["lat","lon","alt_ft"]).T

    #get heading of route
    for index in range(0,len(df)-1,1):
        lat,lon = df.loc[index,["lat","lon"]]
        next_lat,next_lon = df.loc[index+1,["lat","lon"]]
        heading = calculate_initial_compass_bearing(lat,lon,next_lat,next_lon)
        df.loc[index,"heading"] = heading
    df.heading = df.heading.fillna(method="ffill")

    #get flight path angle
    df["gamma"] = (0.3048 * (df["alt_ft"] - df['alt_ft'].shift(1)) /(desired_spacing_km*1000)).fillna(0.)

    #get velocities
    df["V_TAS"] = M*295
    u, v = interpolated_wind_data(df)
    df["V_GS"] = df["V_TAS"] + get_headwind(u,v, df["heading"])

    #flight_path["time"] = flight_path["time"].str.replace("Mon ", "")
    #flight_path["time"] = pd.to_datetime(flight_path["time"], format="%I:%M:%S %p")
    #flight_path.set_index("time", inplace=True)
    #entry,exit = pd.DataFrame(flight_path.iloc[0,:]).T,flight_path.iloc[-1,:].T
    #flight_path = flight_path.resample(str(tau)+'S').ffill(limit=1).interpolate(method='linear').dropna(how="all")
    #flight_path = pd.concat([entry,flight_path])
    #flight_path = flight_path._append(exit)
    #flight_path["V_GS"] = flight_path["speed_kts"] * 0.514444444
    #flight_path["gamma"] = ((0.3048 * (flight_path["alt_ft"] - flight_path['alt_ft'].shift(1)) / tau) / (flight_path["V_GS"])).fillna(0.)
    #u, v = interpolated_wind_data(flight_path)
    #flight_path[["u", "v"]] = np.array([u, v]).T
    #flight_path["headwind"] = get_headwind(flight_path["u"], flight_path["v"], flight_path["heading"])
    #flight_path["V_TAS_est"] = flight_path["V_GS"] - flight_path["headwind"]
    return df