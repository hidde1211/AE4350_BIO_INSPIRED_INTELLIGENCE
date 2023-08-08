import matplotlib.pyplot as plt
import geopandas as gpd
import pandas as pd
from wind_sim import *
from ambiance import Atmosphere
import matplotlib.dates as mdates
import geopy.distance
from flight_mechanics import get_flight_performance

myFmt = mdates.DateFormatter('%H:%M') # here you can format your datetick labels as desired
param_dict = pd.read_excel("aircraft_params.xlsx")
name,M,FL0,FL_max,m0,mf0,mzf,CD0,k,S,TSFC = param_dict.values[0]


def route_plot(lon,lat,U,V,flight_path):
    fig, ax = plt.subplots()
    LON, LAT = np.meshgrid(lon, lat)
    U, V = U[18][2], V[18][2]

    WIND = (U ** 2 + V ** 2) ** 0.5
    im = ax.pcolormesh(LON, LAT, WIND, cmap="jet")
    stride = 10
    ax.quiver(lon[::stride], lat[::stride], U[::stride, ::stride], V[::stride, ::stride], linewidth=0.1,color = "gray")
    ax.scatter(flight_path["lon"], flight_path["lat"],marker = "x", s = 25 , zorder=5,color = "k")
    ax.plot(flight_path["lon"], flight_path["lat"] ,linewidth  = 2 , zorder=5,color = "k")
    cbar = fig.colorbar(im)
    cbar.ax.set_ylabel("wind speed [m/s] at FL362", rotation=90)
    countries = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
    countries.boundary.plot(ax=ax, color="k")
    ax.set_xlim(np.min(lon), np.max(lon))
    ax.set_ylim(np.min(lat), np.max(lat))
    ax.set_ylabel("lat [deg]")
    ax.set_xlabel("lon [deg]")
    ax.set_title("BA115 17-04-2023 resampled route with 50km spacing")

    fig,ax = plt.subplots()
    flight_path["alt_ft"].plot(ax = ax,rot=0)
    ax.set_xlabel("cruise time [H:M]")
    ax.set_ylabel("altitude [ft]")
    fig.suptitle("BA115 altitude profile")
    plt.gca().xaxis.set_major_formatter(myFmt)

    fig,ax = plt.subplots()
    flight_path[["V_TAS","V_GS"]].plot(ax = ax,rot=0)
    ax.set_xlabel("cruise time [H:M]")
    ax.set_ylabel("speed [m/s]")
    fig.suptitle("BA115 speed profile")
    plt.gca().xaxis.set_major_formatter(myFmt)

def env_map(flight_path):
    fig,ax = plt.subplots()
    y = np.arange(flight_path["alt_ft"].min(), 40000 + 100, 100)
    wind_env = pd.DataFrame([],index = y, columns=flight_path.index)
    L_D_env = pd.DataFrame([],index = y, columns=flight_path.index)
    assumed_fuel_burn = 7.88 #kg/km
    m = m0

    for index, coordinates in enumerate(flight_path[["lat", "lon","heading"]].values):
        pos = np.ones((len(y), 3))
        pos[:, :-1] = coordinates[:-1]
        heading = coordinates[-1]
        pos[:, -1] = y
        pos = pd.DataFrame(pos, columns=["lat", "lon", "alt_ft"])

        if index>1:
            old_lat, old_lon = flight_path.iloc[index-1]["lat"], flight_path.iloc[index-1]["lon"]
            new_lat, new_lon = flight_path.iloc[index]["lat"], flight_path.iloc[index]["lon"]
            dx = geopy.distance.geodesic((old_lat, old_lon), (new_lat, new_lon)).km
            m = m - assumed_fuel_burn*dx
            for h in y:
                L_D = get_flight_performance(m,M*295,h,0.)[0]
                L_D_env.loc[h,flight_path.index[index]] = L_D

        u, v = interpolated_wind_data(pos)
        headwind = get_headwind(u,v,heading)
        wind_env.loc[y,flight_path.index[index]] = -headwind

    wind_env = flight_path["V_TAS"].values - wind_env.astype(float)
    L_D_env = L_D_env.astype(float)
    T,Y = np.meshgrid(flight_path.index,y)
    im = ax.pcolormesh(T,Y,wind_env.values,cmap ="jet",vmax=260)
    cbar = fig.colorbar(im)
    ax.set_xlabel("cruise time [H:M]")
    ax.set_ylabel("altitude [ft]")
    ax.set_title(f"Ground speed environment along predefined route (M = {M})")
    cbar.ax.set_ylabel("ground speed [m/s]", rotation=90)
    plt.gca().xaxis.set_major_formatter(myFmt)

    fig,ax = plt.subplots()
    im = ax.pcolormesh(T,Y,L_D_env.values,cmap ="jet")
    cbar = fig.colorbar(im)
    ax.set_xlabel("cruise time [H:M]")
    ax.set_ylabel("altitude [ft]")
    ax.set_title(f"L/D environment along predefined route (mfx = {assumed_fuel_burn} kg/km and M = {M})")
    cbar.ax.set_ylabel("L/D [-]", rotation=90)
    plt.gca().xaxis.set_major_formatter(myFmt)

def wind_plot(lat,lon,U,V,levels):
    times = [16,18,20,22]
    fig,ax = plt.subplots(len(levels),len(times))
    for jindex,time_index in enumerate(times):
        for index,level in enumerate(levels):
            h = int(reverse_ISA(level))
            u,v = U[time_index,index],V[time_index,index]
            wind = np.sqrt(u*u + v*v)
            X,Y = np.meshgrid(lon,lat)
            im = ax[index,jindex].pcolormesh(X,Y,wind,cmap="jet",vmin = 5, vmax = 50)
            stride=10
            ax[index,jindex].quiver(lon[::stride], lat[::stride],u[::stride,::stride], v[::stride,::stride],linewidth = 0.3,color = "gray")
            countries = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
            countries.boundary.plot(ax=ax[index,jindex],color="k")
            ax[index,jindex].set_xlim(np.min(lon),np.max(lon))
            ax[index,jindex].set_ylim(np.min(lat),np.max(lat))
            ax[index,jindex].set_yticks([])
            ax[index,jindex].set_xticks([])
            if jindex == 0:
                ax[index,jindex].set_ylabel(str(h)+"ft", fontsize=18)

            if index == 0:
                ax[index,jindex].set_title(str(time_index)+":00 UTC", fontsize=18)

            plt.subplots_adjust(hspace = 0.1,wspace = 0.1)

    fig.suptitle("Wind data over North Atlantic 17-04-2023",fontsize = 24)
    cbar_ax = fig.add_axes([0.9, 0.1, 0.03, 0.8])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.ax.get_yaxis().labelpad = 15
    cbar.ax.set_ylabel('wind speed [m/s]', rotation=90, fontsize = 18)

def basic_environment_plot(lat,lon,U,V):
    y = np.arange(36000, 40000 + 100, 100)
    N = 200
    lon_course = np.linspace(-0.11,-73.9,N)
    lat_course = 45*np.ones(N)

    course = pd.DataFrame([lat_course,lon_course]).T
    u_env = pd.DataFrame([],index = y, columns= lon_course)
    v_env = pd.DataFrame([],index = y, columns= lon_course)
    rho_env = pd.DataFrame([],index = y, columns= lon_course)
    a_env = pd.DataFrame([],index = y, columns= lon_course)

    for index, coordinates in enumerate(course.values):
        pos = np.ones((len(y), 3))
        pos[:, :-1] = coordinates
        pos[:, -1] = y
        pos = pd.DataFrame(pos, columns=["lat", "lon", "alt_ft"])
        u, v = interpolated_wind_data(pos)
        h_m = y * 0.3048
        rho = Atmosphere(h_m).density
        a = Atmosphere(h_m).speed_of_sound

        u_env.loc[:,lon_course[index]] = u
        v_env.loc[:,lon_course[index]] = v
        rho_env.loc[:,lon_course[index]] = rho
        a_env.loc[:,lon_course[index]] = a

    u_env = u_env.astype(float)
    v_env = v_env.astype(float)
    rho_env = rho_env.astype(float)
    a_env = a_env.astype(float)

    fig,ax = plt.subplots(2,2)
    LON,Y = np.meshgrid(lon_course,y)
    im = ax[0,0].pcolormesh(LON,Y,u_env.values,cmap ="jet")
    cbar = fig.colorbar(im)
    im = ax[0,1].pcolormesh(LON,Y,v_env.values,cmap ="jet")
    cbar = fig.colorbar(im)
    im = ax[1,0].pcolormesh(LON,Y,rho_env.values,cmap ="jet")
    cbar = fig.colorbar(im)
    im = ax[1,1].pcolormesh(LON,Y,a_env.values,cmap ="jet")
    cbar = fig.colorbar(im)

    ax[0,0].set_xlabel("lon [deg]")
    ax[0,1].set_xlabel("lon [deg]")
    ax[1,0].set_xlabel("lon [deg]")
    ax[1,1].set_xlabel("lon [deg]")

    ax[0,0].set_ylabel("altitude [ft]")
    ax[0,1].set_ylabel("altitude [ft]")
    ax[1,0].set_ylabel("altitude [ft]")
    ax[1,1].set_ylabel("altitude [ft]")

    ax[0,0].set_title("U wind component [m/s]")
    ax[0,1].set_title("V wind component [m/s]")
    ax[1,0].set_title("Air density [kg/m3]")
    ax[1,1].set_title("Speed of sound [m/s]")

    fig.suptitle("Environment at constant 45 deg latitude")
    plt.subplots_adjust(hspace=0.35,wspace=0.35)

def perf_plot(dfs,labels):
    fig, ax = plt.subplots(2, 2, sharex=True)
    for index,flight_path in enumerate(dfs):
        plt.gca().xaxis.set_major_formatter(myFmt)
        flight_path["alt_ft"].plot(ax=ax[0, 0],label=labels[index])
        flight_path["V_GS"].plot(ax=ax[1, 0],label=labels[index])
        (1000*flight_path["mfx"]).plot(ax=ax[0, 1],label=labels[index])
        flight_path["L_D"].plot(ax=ax[1, 1],label=labels[index])
        ax[0,0].set_ylabel("altitude [ft]")
        ax[0,0].legend()
        ax[0,1].set_ylabel("fuel flow [kg/km]")
        ax[0,1].legend()
        ax[1,0].set_ylabel("ground speed [m/s]")
        ax[1,0].legend()
        ax[1,1].set_ylabel("lift-to-drag ratio [-]")
        ax[1,1].legend()

def RL_perf_plot(perf_df,m_benchmark,t_benchmark):
    fig,ax = plt.subplots()
    ax.plot(np.arange(1,len(perf_df)+1,1),perf_df["mf"].values)
    ax.plot(np.arange(1,len(perf_df)+1,1),m_benchmark*np.ones(len(perf_df)),label='benchmark')
    ax.set_xlabel("Number of episodes")
    ax.set_ylabel("Fuel used [kg]")
    ax.legend()

    fig,ax = plt.subplots()
    ax.plot(np.arange(1,len(perf_df)+1,1),perf_df["T"].values/60)
    ax.plot(np.arange(1,len(perf_df)+1,1),t_benchmark*np.ones(len(perf_df))/60,label='benchmark')
    ax.set_xlabel("Number of episodes")
    ax.set_ylabel("Cruise time [min]")
    ax.legend()

def stat_plot(stat_df,parameter, rl_params):
    #RL Parameters
    epsilon, N_ep,alpha ,gamma_rl = list(rl_params.values())

    fig,ax = plt.subplots(2,2)
    stat_df["av mf"].plot(ax=ax[0,0])
    ax[0,0].set_xlabel(parameter)
    stat_df["std mf"].plot(ax=ax[1, 0])
    ax[1,0].set_xlabel(parameter)
    stat_df["av T"].plot(ax=ax[0,1])
    ax[0,1].set_xlabel(parameter)
    stat_df["std T"].plot(ax=ax[1, 1])
    ax[1,1].set_xlabel(parameter)
    plt.show()

