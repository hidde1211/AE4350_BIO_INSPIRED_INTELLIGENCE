import numpy as np
import pandas as pd
from wind_sim import *
from ambiance import Atmosphere

param_dict = pd.read_excel("aircraft_params.xlsx")
name,M,FL0,FL_max,m0,mf0,mzf,CD0,k,S,TSFC = param_dict.values[0]

def get_flight_performance(m,V,h,gamma):
    # GET ATMOSPHERIC STATE
    h_m = h * 0.3048
    rho = Atmosphere(h_m).density[0]

    CL = m * 9.81 / (0.5 * rho * V * V * S)
    CD = CD0 + k * CL ** 2

    Tr = np.maximum(CD * 0.5 * rho * V * V * S + m * 9.81 * gamma, 0.)
    mf_dot = TSFC * Tr / (1 * 10 ** 6)

    return CL/CD , mf_dot

def const_flight(flight_path,dx):
    flight_path[["m","mf","L_D","mfx","t"]] = m0,mf0,np.nan,np.nan,np.nan

    for index in range(len(flight_path)):
        #GET AIRCRAFT STATE
        m,mf,Vg,V,h,gamma = flight_path.loc[index,["m","mf","V_GS","V_TAS", "alt_ft", "gamma"]]
        L_D,mf_dot = get_flight_performance(m,V,h,gamma)
        mfx = mf_dot/Vg

        flight_path.loc[index, ["L_D", "mfx","t"]] = L_D, mfx, 1000*dx/Vg
        if index<len(flight_path)-1:
            flight_path.loc[index+1,["m","mf"]] = m - mfx*1000*dx, mf - mfx*1000*dx

    flight_path.index = pd.to_datetime(flight_path["t"].cumsum(),unit='s')
    return flight_path







