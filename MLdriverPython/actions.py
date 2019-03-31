import library as lib
import numpy as np
import steering_lib
import agent
import matplotlib.pyplot as plt
import copy

import predict_lib
def comp_steer(nets,state):

    return steer

def comp_max_acc(accNet,state_vehicle_values,dsteer,acc_pos,planPar):
    X = state_vehicle_values+[dsteer]+[planPar.max_roll]
    return accNet.get_Y(X)

class TargetPoint:#target point - [dx,dy, velocity],
    abs_pos = []
    rel_pos = []
    vel = 0

def print_stateVehicle(stateVehicle):
    print("abs_pos:",stateVehicle.abs_pos)
    print("rel_pos:",stateVehicle.rel_pos)
    print("values:",stateVehicle.values)

def comp_action(nets,state,trainHP,targetPoint):#compute the actions (and path) given the target point(s)
    #rel_pos of the vehicle - relative to last position (at last time step)
    #rel_pos of targetPoint - relative to the vehicle
    #abs_pos - relative to the initial pos, a new one at every time step.
    fig,[ax_abs,ax_rel] = plt.subplots(2)
    ax_rel.axis('equal')
    ax_abs.axis('equal')
    ax_abs.plot([targetPoint.abs_pos[0]],[targetPoint.abs_pos[1]],'x')

    StateVehicle = state.Vehicle
    ax_abs.plot([StateVehicle.abs_pos[0]],[StateVehicle.abs_pos[1]],'o')

    min_vel = 0.1

    acc = 1.0
    steer = state.Vehicle.values[trainHP.vehicle_ind_data['steer']]
    #acc_pos = True
    dist_to_target = np.linalg.norm(state.Vehicle.rel_pos - targetPoint.rel_pos)# need to be np array 
    last_dist_to_target = dist_to_target+1.0 #to ensure that last_dist_to_target is bigger then dist_to_target
    #vehicle_states = []
    print_stateVehicle(StateVehicle)

    while dist_to_target < last_dist_to_target:#not always the right condition
    #for i in range(15):
        last_dist_to_target = dist_to_target
        
        
        dsteer = steering_lib.comp_steer(nets,StateVehicle,targetPoint,trainHP)
        steer=dsteer
        print("steer:",steer)
        steer = np.clip(steer,-0.5,0.5)
        
        StateVehicle = step(StateVehicle,acc,steer,nets.TransNet,trainHP)
        print_stateVehicle(StateVehicle)
        targetPoint = target_to_vehicle(targetPoint,StateVehicle)#update the relative to the vehicle position of the target 

        if StateVehicle.values[0] < min_vel:
            break

        ax_abs.plot([StateVehicle.abs_pos[0]],[StateVehicle.abs_pos[1]],'o')
        ax_rel.plot([targetPoint.rel_pos[0]],[targetPoint.rel_pos[1]],'x')

        dist_to_target = np.linalg.norm(state.Vehicle.rel_pos - targetPoint.rel_pos)
    #    acc = comp_max_acc(nets.accNet,state.Vehicle.values,dsteer)
    #    stateVehicleNext = get_next_vehicle_state(nets.transNet,state.Vehicle,dsteer,acc)
    #    state.vehicle = stateVehicleNext
    #    acc_pos = False
        acc = -1.0

    #ax_abs.set_ylim(0,15)
    #ax_abs.set_xlim(-7.5,7.5)

    #acc = 1.0
    #steer = -0.3 
    #vehicle_states = []
    #for i in range(10):
    #    StateVehicle = step(StateVehicle,acc,steer,nets.TransNet,trainHP)
    #    targetPoint = target_to_vehicle(targetPoint,StateVehicle)#update the relative to the vehicle position of the target 

    #    ax_abs.plot([StateVehicle.abs_pos[0]],[StateVehicle.abs_pos[1]],'o')
    #    ax_rel.plot([targetPoint.rel_pos[0]],[targetPoint.rel_pos[1]],'x')
    plt.show()
    return #acc, steer


def step(stateVehicle,acc,steer,TransNet,trainHP):#get a state and actions, return next state
    x = [stateVehicle.values+[acc,steer]]
    print("x:",x)
    y = TransNet.predict(np.array(x))[0]
    nextState = agent.State()
    delta_values = y[:len(trainHP.vehicle_ind_data)].tolist()
    nextState.Vehicle.values = [stateVehicle.values[i]+delta_values[i] for i in range(len(delta_values))]
    if nextState.Vehicle.values[0] > 20:
        nextState.Vehicle.values[0] = 20
    nextState.Vehicle.rel_pos = y[len(trainHP.vehicle_ind_data):len(trainHP.vehicle_ind_data)+2]
    nextState.Vehicle.rel_ang = y[len(trainHP.vehicle_ind_data)+2:]

    nextState.Vehicle.abs_pos,nextState.Vehicle.abs_ang = predict_lib.comp_abs_pos_ang(nextState.Vehicle.rel_pos,nextState.Vehicle.rel_ang,
                                                                                       stateVehicle.abs_pos,stateVehicle.abs_ang)
    #nextState.Vehicle.abs_pos = stateVehicle.abs_pos + nextState.Vehicle.rel_pos

    #nextState.Vehicle.abs_ang = stateVehicle.abs_ang + nextState.Vehicle.rel_ang

    return nextState.Vehicle

def target_to_vehicle(targetPoint,nextStateVehicle):#update the position of the target point relative to the new position of the vehicle
    newTargetPoint = TargetPoint()
    rel_pos = lib.to_local(targetPoint.abs_pos,nextStateVehicle.abs_pos,nextStateVehicle.abs_ang)
    newTargetPoint = copy.copy(targetPoint)
    newTargetPoint.rel_pos = np.array(rel_pos)
    return newTargetPoint