import predict_lib
import numpy as np
import math
#state = [velocity, current steering, roll]
def get_dsteer_max(SteerNet,state_vehicle,acc, direction):#get maximum change in steering in a given state
    current_roll = state_vehicle[2]
    des_roll = -0.03*direction
    #droll = des_roll - current_roll
    print("droll:",des_roll)
    dsteer_max = SteerNet.predict(np.array([state_vehicle+[acc,des_roll]]))[0][0]
    return dsteer_max

def get_zero_dsteer(state, dsteer_max,net):#compute change in steering in a given state that zeroes the steering. maximum if the next state does not reach it
    next_state
    dsteer
    return dsteer

def look_ahead(state,dsteer,net):#predict states from current state and given dsteer, policy: first step - dsteer, next steps zero dsteer. 
                                 #stop if distance to target point at the next step is smaller than at the current step
    return state_vec

def distance_to_target(pos1,pos2,target):

    return distance, dir

def comp_steer_direction(targetPoint):
    
    return math.copysign(1.0, -targetPoint.rel_pos[0])

def comp_steer(nets,StateVehicle,targetPoint,trainHP,dir = dir):#search the steering 
    #return dsteer such that applying it the first step and then zeroing the steering results at the minimum distance from the target.
    #first guess the maximum dsteer - if distance is positive (cannot reach the point at none steering) return. 
    dir = comp_steer_direction(targetPoint)
    acc = 1.0
    dsteer_max = get_dsteer_max(nets.SteerNet,StateVehicle.values,acc,dir)
    
    dsteer = dsteer_max
    return dsteer