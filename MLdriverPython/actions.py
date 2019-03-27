import library as lib
import numpy as np
import steering_lib
import agent

def comp_steer(nets,state):

    return steer

def comp_max_acc(accNet,state_vehicle_values,dsteer,acc_pos,planPar):
    X = state_vehicle_values+[dsteer]+[planPar.max_roll]
    return accNet.get_Y(X)

class TargetPoint:#target point - [dx,dy, velocity],
    abs_pos = []
    rel_pos = []
    vel = 0

def comp_action(nets,state,trainHP,targetPoint):#compute the actions (and path) given the target point(s)
    acc_pos = True
    dist_to_target = np.linalg.norm(state.Vehicle.rel_pos, state.targetPoint.rel_pos)# need to be np array 
    last_dist_to_target += dist_to_target+1.0 #to ensure that last_dist_to_target is bigger then dist_to_target
    vehicle_states = []
    while dist_to_target < last_dist_to_target:#not always the right condition
        last_dist_to_target = dist_to_target
        dist_to_target = np.linalg.norm(state.Vehicle.rel_pos, state.targetPoint.rel_pos)
        
        dsteer = steering_lib.comp_steer(nets,state,targetPoint,trainHP)

        acc = comp_max_acc(nets.accNet,state.Vehicle.values,dsteer)
        stateVehicleNext = get_next_vehicle_state(nets.transNet,state.Vehicle,dsteer,acc)
        state.vehicle = stateVehicleNext
        acc_pos = False

    return acc, steer


def step(stateVehicle,acc,steer,TransNet,trainHP):#get a state and actions, return next state
    x = [[stateVehicle.values]+[acc,steer]]
    y = TransNet.predict(np.array(x))
    nextStateVehicle = agent.State()
    nextStateVehicle.Vehicle.values[:len(trainHP.vehicle_ind_data)]
