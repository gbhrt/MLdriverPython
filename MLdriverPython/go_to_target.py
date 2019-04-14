import matplotlib.pyplot as plt
import numpy as np
import time
import copy
import random

import actions
import target_point
import agent
import hyper_parameters
import library as lib

def get_random_state():
    S = agent.State()
    vel = random.uniform(0,30)
    steer = random.uniform(-0.7,0.7)
    roll = random.uniform(-0.07,0.07)
    S.Vehicle.values = [vel,steer,roll]#v,steer,roll

    S.Vehicle.abs_pos = np.array([0.0,0.0])
    S.Vehicle.abs_ang = 0.0

    return S.Vehicle

def get_random_target(StateVehicle):
    targetPoint = target_point.TargetPoint()
    #targetPoint.abs_pos = np.array([-5.0,7.0]) #x,y
    #targetPoint = actions.target_to_vehicle(targetPoint,S.Vehicle)

    

    #x = 10
    #y = 10
    #targetPoint.abs_pos = np.array([x,y]) #x,y
    #targetPoint = actions.comp_rel_target(targetPoint,S.Vehicle)
    targetPoint.rel_pos = [random.uniform(-30,30),random.uniform(0,30)]
    targetPoint = actions.comp_abs_target(targetPoint,StateVehicle) 
    targetPoint.vel = random.uniform(0,30)
    return targetPoint

def go_to_target(Agent,StateVehicle,targetPoint,stop_flag = False,ax =None):
    StateVehicle_vec = [copy.deepcopy(StateVehicle)]
    reachable_flag = False
    while targetPoint.rel_pos[1] > 0:
        t = time.clock()
        acc,steer,_,_ = actions.comp_action(Agent.nets,StateVehicle,Agent.trainHP,targetPoint,stop_flag,ax)
        print("comp actions time:",time.clock() - t)
        StateVehicle = actions.step(StateVehicle,acc,steer,Agent.nets.TransNet,Agent.trainHP)
        targetPoint = actions.comp_rel_target(targetPoint,StateVehicle)

        StateVehicle_vec.append(copy.deepcopy(StateVehicle))
    return acc,steer,StateVehicle_vec,reachable_flag 

def learn_actions_supervised():
    plot_states_flag = True


    HP = hyper_parameters.ModelBasedHyperParameters()
    Agent = agent.Agent(HP)

    episodes_num = 10
    random.seed(None)#temp
    
    for i in range(episodes_num):
        fig,ax = plt.subplots(1)
        ax.axis('equal')
        stop_flag = False

        StateVehicle = get_random_state()
        actions.print_stateVehicle(StateVehicle)
        targetPoint =  get_random_target(StateVehicle)
        print("targetPoint:", targetPoint.rel_pos,"vel:", targetPoint.vel)

        acc,steer,StateVehicle_vec,reachable_flag = go_to_target(Agent,StateVehicle,targetPoint,stop_flag,ax)

        if plot_states_flag: 
            actions.plot_target(targetPoint,ax)
            actions.plot_state_vec(StateVehicle_vec,ax)
            plt.show()


if __name__ == "__main__":
    learn_actions_supervised()
