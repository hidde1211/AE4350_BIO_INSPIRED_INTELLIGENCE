import pandas as pd
from wind_sim import *
import random
from flight_mechanics import get_flight_performance
import warnings
import geopy.distance

warnings.simplefilter(action='ignore')
param_dict = pd.read_excel("aircraft_params.xlsx")
name,M,FL0,FL_max,m0,mf0,mzf,CD0,k,S,TSFC = param_dict.values[0]

def get_new_state(state,old_state,action,dy,dx):
    #STEP 1 NEW ALTITUDE WITHIN BOUNDS
    climb = 0

    if action == 1:
        climb = dy

    if action == 2:
        climb = - dy

    state["alt_ft"] = old_state["alt_ft"] + climb

    if state["alt_ft"] > FL_max*100:
        state["alt_ft"] = FL_max*100

    if state["alt_ft"] < FL0*100:
        state["alt_ft"] = FL0*100

    state["gamma"] = 0.3048*(1/1000)*climb/dx
    state["m"] = old_state["m"] - 1000*dx*old_state["mfx"]
    state["mf"] = old_state["mf"] - 1000*dx*old_state["mfx"]
    state["t"] = old_state["t"] + 1000*dx/old_state["V_GS"]
    return state

def get_objective(state):
    #SPEED CALCULATIONS
    u,v = interpolated_wind_data(state)
    headwind = get_headwind(u,v,state["heading"])[0]
    state["headwind"] = headwind
    state.loc["V_GS"] = state.loc["V_TAS"] + state.loc["headwind"]

    #FLIGHT MECHANICS CALCULATIONS
    L_D, mf_dot = get_flight_performance(state["m"],state["V_TAS"],state["alt_ft"],state["gamma"])
    mfx = mf_dot/state.loc["V_GS"]
    state[["L_D", "mfx"]] = L_D, mfx
    return state

def statistical_perf(RL_perf):
    mu,std = RL_perf.iloc[20:,].mean().values,RL_perf.iloc[20:,].std().values
    stat = np.array([mu,std]).flatten()
    stat[[1,-1]] = stat[[1,-1]]/60
    stat = pd.DataFrame(stat,index = ["av mf","av T","std mf","std T"]).T
    return stat

def reinforcement_learning_flight_path(flight_path,rl_params,mf_bench,t_bench,dx):
    #RL Parameters
    epsilon, N_ep,alpha ,gamma_rl = list(rl_params.values())

    #Define environment parameters
    dy = 100
    end_alt = flight_path["alt_ft"].iloc[-1]

    y = np.arange(FL0*100,FL_max*100+dy,dy)
    Nhor,Nver = len(flight_path),len(y)

    #Possible actions
    STRAIGHT = 0
    UP = 1
    DOWN = 2
    action_list = np.array([STRAIGHT,UP,DOWN])

    #Value function and Policy matrix
    Q = np.zeros((Nver,Nhor,len(action_list)))
    policy = np.random.randint(0, len(action_list), size=(Nver, Nhor))

    Rl_perf = pd.DataFrame([],index = np.arange(1,N_ep+1,1),columns = ["mf","T"])
    for ep in range(N_ep):
        print(f"episiode nr {ep+1}.....")
        state_history = flight_path[["lat","lon","heading","alt_ft","V_TAS"]].reset_index()
        state_history.loc[np.arange(1,len(state_history)-1,1),"alt_ft"] = np.nan
        state_history.loc[:,['gamma','headwind','V_GS','m','mf','t','L_D','mfx']] = np.nan
        state_history.loc[0,["gamma","m","mf","t"]] = 0.,m0,mf0,0.
        state_history.loc[0,:] = get_objective(state_history.loc[0,:])

        actionIdx = policy[0,0]
        action = action_list[actionIdx]

        for k in range(1,Nhor):
            #UPDATE TO NEW STATE
            state = state_history.loc[k-1,:]
            state_idx = np.argwhere(y == state["alt_ft"])[0][0]

            new_state = state_history.loc[k,:]
            new_state = get_new_state(new_state,state,action,dy,dx)
            new_state = get_objective(new_state)
            state_history.loc[k,:] = new_state
            newstate_idx = np.argwhere(y == new_state["alt_ft"])[0][0]

            #GET STATE REWARD
            if new_state["L_D"] > state["L_D"]:
                reward = 1

            else:
                reward = -1

            if new_state["V_GS"] > state["V_GS"]:
                reward = reward + 1

            else:
                reward = reward -1

            if k == Nhor -1 :
                t = state_history.loc[k,"t"]
                mf = mf0 - state_history.loc[k,"mf"]

                if np.isnan(Rl_perf["mf"].min()) == False and np.isnan(Rl_perf["T"].min()) == False:
                    mf_benchmark = np.minimum(mf_bench,Rl_perf["mf"].min())
                    T_benchmark = np.minimum(t_bench,Rl_perf["T"].min())

                else:
                    mf_benchmark = mf_bench
                    T_benchmark = t_bench

                if t<= T_benchmark:
                    reward = reward + Nhor
                else:
                    reward = reward - Nhor

                if mf<= mf_benchmark:
                    reward = reward + Nhor
                else:
                    reward = reward - Nhor

            #GET NEW ACTION BASED ON POLICY
            newActionIdx = policy[newstate_idx,k]
            newAction = action_list[newActionIdx]

            #UPDATE VALUE FUNCTION BASED ON NEW STATE AND NEW ACTION
            current_Q = Q[state_idx, k-1, actionIdx]
            next_Q = Q[newstate_idx, k, newActionIdx]

            #SARSA VALUE FUNCTION UPDATE RULE
            Q[state_idx, k-1, actionIdx] = current_Q + alpha*(reward + gamma_rl*(next_Q)-current_Q)

            #SARSA POLICY FUNCTION UPDATE RULE
            rand = random.uniform(0,1)

            if rand < epsilon:
                policy[state_idx,k-1] = np.random.randint(0,len(action_list))

            else:
                max_Q_policy = np.max(Q[state_idx,k-1])
                max_Q_arg = np.argwhere(Q[state_idx,k-1] == max_Q_policy)
                policy[state_idx,k-1] = max_Q_arg[0]

            actionIdx = newActionIdx
            action = newAction

        Rl_perf.loc[ep+1,"mf"] = mf0 - state_history.iloc[-1]["mf"]
        Rl_perf.loc[ep+1,"T"] = state_history.iloc[-1]["t"]
    return state_history,Rl_perf