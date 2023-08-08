import matplotlib.pyplot as plt
from wind_sim import *
from plotting import *
import pandas as pd

#BOEING 777-300ER Cruise Flight Parameters
M = 0.83                #cruise mach number [-]
CD0,k =  0.013,0.047    #drag polar
S =  436.8              #surface area [m2]
TSFC = 16               #thrust specific fuel consumption druing cruise [g/kNs]
FL0 = 360               #initial cruise flight level
FL_max = 400            #maximum cruise flight level
m_tow = 351530          #maximum take-off weight [kg]
m_oew = 167800          #operating empty weight [kg]
N_pax = 400             #number of passengers
seat_p = 0.8            #percentage of occupied seats
m_pax = N_pax*seat_p*100#payload weight (every passenger carries 100kg)
m_zfw = m_oew + m_pax   #zero fuel weight
mf0 = 0.9*60000         #fuel weight based on flight planning LHR-JFK [kg] when entering North Atlantic Track
m0 = m_zfw + mf0    #begin mass when entering cruise

aircraft_params = {"name":"B777-300ER","M":M,
                   "FL0":FL0,"FL_max":FL_max,
                   "m0":m0,
                   "mf0":mf0,"mzf":m_zfw,"CD0":CD0,"k":k,"S":S,"TSFC":TSFC}
param_df = pd.DataFrame(aircraft_params,index=[0])
param_df.to_excel("aircraft_params.xlsx",index=False)

#WIND ENVIRONMENT DATA
import netCDF4 as nc
ds = nc.Dataset("weather_data_17_04_2023.nc")
lon,lat,V,U,levels,times = ds.variables['longitude'][:],ds.variables['latitude'][:],ds.variables['v'][:],ds.variables['u'][:],ds.variables['level'][:],ds.variables['time'][:]
basic_environment_plot(lat,lon,U,V)

#BA115 (LHR-JFK) BENCHMARK FLIGHT PATH
from flight_data import get_flight_path_data
from flight_mechanics import *
spacing_km = 50.0
flight_path = get_flight_path_data(spacing_km)
flight_path = const_flight(flight_path,spacing_km)
route_plot(lon,lat,U,V,flight_path)
env_map(flight_path)

#get benchmarks
mf_bench = mf0 - flight_path.iloc[-1,:]["mf"]
t_bench = (flight_path.index.values[-1] - flight_path.index.values[0])/np.timedelta64(1, 's')

#Reinfocement Learning
epsilon = 0.2    #exploration rate
N_ep =  150      #number of episodes
alpha = 0.5      #learning rate
gamma_rl = 0.9   #discount rate

rl_params = {"epsilon":epsilon,"N_ep":N_ep,"alpha":alpha,"gamma":gamma_rl}

#Reinforcement Learning Flight Path
from reinforcement_learning import *
RL_run = False

if RL_run:
    RL_flight_path,RL_perf = reinforcement_learning_flight_path(flight_path,rl_params,mf_bench,t_bench,spacing_km)
    RL_stat = statistical_perf(RL_perf)

    RL_flight_path["time"] = pd.to_datetime(flight_path.index.values[0]) + pd.to_timedelta(RL_flight_path['t'], unit='s')
    RL_flight_path.set_index("time",inplace=True)

    perf_plot([RL_flight_path,flight_path],["RL-based flight","conventional BA115"])
    RL_perf_plot(RL_perf,mf_bench,t_bench)

#SENSITIVITY ANALYSIS
N_ep_range = [50,100,1000]
val_range = [0.,0.4,0.6,1.]

#Default reinfocement learning params
epsilon_0 = 0.2    #exploration rate
N_ep_0 =  100      #number of episodes
alpha_0 = 0.5      #learning rate
gamma_rl_0 = 0.9   #discount rate

sens_analysis = True

if sens_analysis:
    N_df = pd.DataFrame([])
    for epsilon in val_range:
        rl_params = {"epsilon": epsilon, "N_ep": N_ep_0, "alpha": alpha_0, "gamma": gamma_rl_0}
        RL_flight_path, RL_perf = reinforcement_learning_flight_path(flight_path, rl_params, mf_bench, t_bench,spacing_km)
        RL_stat = statistical_perf(RL_perf)
        N_df = pd.concat([N_df,RL_stat])

    N_df = N_df.astype(float)
    fig,ax = plt.subplots(1,2)
    ax[0].plot(val_range,N_df["av mf"],c="tab:orange",linestyle = "--",label = "steady state average")
    ax[0].fill_between(val_range, (N_df["av mf"]-N_df["std mf"]).values,(N_df["av mf"]+N_df["std mf"]).values,color="tab:blue",alpha = 0.6,label = "steady state average +- standard deviation")
    ax[0].hlines(mf_bench,val_range[0],val_range[-1],color="tab:green",label="benchmark")
    ax[0].legend()
    ax[0].set_ylabel("fuel mass used [kg]")
    ax[0].set_xlabel("epsilon")

    ax[1].plot(val_range,N_df["av T"],c="tab:orange",linestyle = "--",label = "steady state average")
    ax[1].fill_between(val_range, (N_df["av T"]-N_df["std T"]).values,(N_df["av T"]+N_df["std T"]).values,color="tab:blue",alpha = 0.6,label = "steady state average +- standard deviation")
    ax[1].hlines(t_bench/60,val_range[0],val_range[-1],color="tab:green",label="benchmark")
    ax[1].legend()
    ax[1].set_ylabel("cruise time [min]")
    ax[1].set_xlabel("epsilon")
    fig.suptitle("Epsilon sensitivity analysis")

    N_df = pd.DataFrame([])
    for gamma_rl in val_range:
        rl_params = {"epsilon": epsilon_0, "N_ep": N_ep_0, "alpha": alpha_0, "gamma": gamma_rl}
        RL_flight_path, RL_perf = reinforcement_learning_flight_path(flight_path, rl_params, mf_bench, t_bench,spacing_km)
        RL_stat = statistical_perf(RL_perf)
        N_df = pd.concat([N_df,RL_stat])

    N_df = N_df.astype(float)
    fig,ax = plt.subplots(1,2)
    ax[0].plot(val_range,N_df["av mf"],c="tab:orange",linestyle = "--",label = "steady state average")
    ax[0].fill_between(val_range, (N_df["av mf"]-N_df["std mf"]).values,(N_df["av mf"]+N_df["std mf"]).values,color="tab:blue",alpha = 0.6,label = "steady state average +- standard deviation")
    ax[0].hlines(mf_bench,val_range[0],val_range[-1],color="tab:green",label="benchmark")
    ax[0].legend()
    ax[0].set_ylabel("fuel mass used [kg]")
    ax[0].set_xlabel("gamma")

    ax[1].plot(val_range,N_df["av T"],c="tab:orange",linestyle = "--",label = "steady state average")
    ax[1].fill_between(val_range, (N_df["av T"]-N_df["std T"]).values,(N_df["av T"]+N_df["std T"]).values,color="tab:blue",alpha = 0.6,label = "steady state average +- standard deviation")
    ax[1].hlines(t_bench/60,val_range[0],val_range[-1],color="tab:green",label="benchmark")
    ax[1].legend()
    ax[1].set_ylabel("cruise time [min]")
    ax[1].set_xlabel("gamma")

    fig.suptitle("Gamma sensitivity analysis")

    N_df = pd.DataFrame([])
    for alpha in val_range:
        rl_params = {"epsilon": epsilon_0, "N_ep": N_ep_0, "alpha": alpha, "gamma": gamma_rl_0}
        RL_flight_path, RL_perf = reinforcement_learning_flight_path(flight_path, rl_params, mf_bench, t_bench,spacing_km)
        RL_stat = statistical_perf(RL_perf)
        N_df = pd.concat([N_df,RL_stat])

    N_df = N_df.astype(float)
    fig,ax = plt.subplots(1,2)
    ax[0].plot(val_range,N_df["av mf"],c="tab:orange",linestyle = "--",label = "steady state average")
    ax[0].fill_between(val_range, (N_df["av mf"]-N_df["std mf"]).values,(N_df["av mf"]+N_df["std mf"]).values,color="tab:blue",alpha = 0.6,label = "steady state average +- standard deviation")
    ax[0].hlines(mf_bench,val_range[0],val_range[-1],color="tab:green",label="benchmark")
    ax[0].legend()
    ax[0].set_ylabel("fuel mass used [kg]")
    ax[0].set_xlabel("alpha")

    ax[1].plot(val_range,N_df["av T"],c="tab:orange",linestyle = "--",label = "steady state average")
    ax[1].fill_between(val_range, (N_df["av T"]-N_df["std T"]).values,(N_df["av T"]+N_df["std T"]).values,color="tab:blue",alpha = 0.6,label = "steady state average +- standard deviation")
    ax[1].hlines(t_bench/60,val_range[0],val_range[-1],color="tab:green",label="benchmark")
    ax[1].legend()
    ax[1].set_ylabel("cruise time [min]")
    ax[1].set_xlabel("alpha")

    fig.suptitle("Alpha sensitivity analysis")

    N_df = pd.DataFrame([])
    for N_ep in N_ep_range:
        rl_params = {"epsilon": epsilon_0, "N_ep": N_ep, "alpha": alpha_0, "gamma": gamma_rl_0}
        RL_flight_path, RL_perf = reinforcement_learning_flight_path(flight_path, rl_params, mf_bench, t_bench,spacing_km)
        RL_stat = statistical_perf(RL_perf)
        N_df = pd.concat([N_df,RL_stat])

    N_df = N_df.astype(float)
    fig,ax = plt.subplots(1,2)
    ax[0].plot(np.array(N_ep_range,dtype=int),N_df["av mf"],c="tab:orange",linestyle = "--",label = "steady state average")
    ax[0].fill_between(np.array(N_ep_range,dtype=int), (N_df["av mf"]-N_df["std mf"]).values,(N_df["av mf"]+N_df["std mf"]).values,color="tab:blue",alpha = 0.6,label = "steady state average +- standard deviation")
    ax[0].hlines(mf_bench,N_ep_range[0],N_ep_range[-1],color="tab:green",label="benchmark")
    ax[0].legend()
    ax[0].set_xscale("log")
    ax[0].set_ylabel("fuel mass used [kg]")
    ax[0].set_xlabel("log (Number of episodes)")

    ax[1].plot(np.array(N_ep_range,dtype=int),N_df["av T"],c="tab:orange",linestyle = "--",label = "steady state average")
    ax[1].fill_between(np.array(N_ep_range,dtype=int), (N_df["av T"]-N_df["std T"]).values,(N_df["av T"]+N_df["std T"]).values,color="tab:blue",alpha = 0.6,label = "steady state average +- standard deviation")
    ax[1].hlines(t_bench/60,N_ep_range[0],N_ep_range[-1],color="tab:green",label="benchmark")
    ax[1].legend()
    ax[1].set_xscale("log")
    ax[1].set_ylabel("cruise time [min]")
    ax[1].set_xlabel("log (Number of episodes)")

    fig.suptitle("Number of episodes sensitivity analysis")

    plt.show()















