import library as lib
import numpy as np
import steering_lib
import agent
import matplotlib.pyplot as plt
import copy
import math 

import predict_lib

#debug settings:
print_flag = False
plot_local_steer_comp_flag = False
plot_action_comp_flag = False
plot_states_flag = True
pause_by_user_flag = False

#fig,[ax_abs,ax_rel] = plt.subplots(2)
#ax_rel.axis('equal')
fig,ax_abs = plt.subplots(1)
ax_abs.axis('equal')
if plot_states_flag or plot_action_comp_flag or plot_local_steer_comp_flag:
    plt.ion()





def print_stateVehicle(StateVehicle):
    print("abs_pos:",StateVehicle.abs_pos,"abs_ang:",StateVehicle.abs_ang,"\n values:",StateVehicle.values)

def plot_state(StateVehicle,line = None):
    if line is None:
        line, = ax_abs.plot([StateVehicle.abs_pos[0]],[StateVehicle.abs_pos[1]],'.')

    else:
        ax_abs.plot([StateVehicle.abs_pos[0]],[StateVehicle.abs_pos[1]],'.',color = line.get_color())
    ax_abs.plot([StateVehicle.abs_pos[0],StateVehicle.abs_pos[0]+math.sin(StateVehicle.abs_ang)],[StateVehicle.abs_pos[1],StateVehicle.abs_pos[1]+math.cos(StateVehicle.abs_ang)],color = line.get_color())
    
def draw_state(StateVehicle,line = None):
    plot_state(StateVehicle,line = line)
    plt.draw()
    plt.pause(0.0001)
    return line

def plot_target(targetPoint):
    ax_abs.plot([targetPoint.abs_pos[0]],[targetPoint.abs_pos[1]],'x')
def draw_target(targetPoint):
    plot_target(targetPoint)
    plt.draw()
    plt.pause(0.0001)

def plot_state_vec(StateVehicle_vec):
    for StateVehicle in StateVehicle_vec:
        plot_state(StateVehicle)





def comp_rel_target(targetPoint,StateVehicle):#update the position of the target point relative to the new position of the vehicle
    newTargetPoint = copy.copy(targetPoint)
    rel_pos = lib.to_local(targetPoint.abs_pos,StateVehicle.abs_pos,StateVehicle.abs_ang)
    newTargetPoint.rel_pos = np.array(rel_pos)
    return newTargetPoint

def comp_abs_target(targetPoint,InitStateVehicle):#update the position of the target point relative to the new position of the vehicle
    newTargetPoint = copy.copy(targetPoint)
    abs_pos = lib.to_global(targetPoint.rel_pos,InitStateVehicle.abs_pos,InitStateVehicle.abs_ang)
    newTargetPoint.abs_pos = np.array(abs_pos)
    return newTargetPoint

def comp_max_acc(accNet,state_vehicle_values,dsteer,acc_flag,planPar):
    X = state_vehicle_values+[dsteer]+[planPar.max_roll]
    return accNet.get_Y(X)

def get_dsteer_max(SteerNet,state_vehicle,acc, direction,trainHP):#get maximum change in steering in a given state
    current_roll = state_vehicle[2]
    des_roll = -trainHP.plan_roll*direction
    #droll = des_roll - current_roll
    #print("droll:",des_roll)
    steer_max = SteerNet.predict(np.array([state_vehicle+[acc,des_roll]]))[0][0]
    return np.clip(steer_max,-0.7,0.7).item()

def comp_steer_direction(targetPoint):
    return math.copysign(1.0, -targetPoint.rel_pos[0])

def comp_distance_from_target_after_zeroing(nets,StateVehicle,targetPoint,acc,steer,trainHP,stop_flag):#get state and actions, return the x of relative distance to target point
    #col=np.random.rand(3,)
    StateVehicle = step(StateVehicle,acc,steer,nets.TransNet,trainHP)
    #StateVehicle1 = step(StateVehicle,1.0,steer,nets.TransNet,trainHP)
    if print_flag: print("******compute zeroing**********")

    if plot_local_steer_comp_flag: line = draw_state(StateVehicle)

    if print_flag: print_stateVehicle(StateVehicle)
    steer = 0.0
    acc = -1.0#0.0
    cnt = 0
    while abs(StateVehicle.values[1]) > 0.01 and targetPoint.rel_pos[1]>0 and StateVehicle.values[0] > 0 and not stop_flag:#steering > 0 and target in front of the vehicle
        #print("while - comp_distance_from_target_after_zeroing. stop_flag:",stop_flag)
        StateVehicle = step(StateVehicle,acc,steer,nets.TransNet,trainHP)
        if plot_local_steer_comp_flag: draw_state(StateVehicle,line = line)

        if print_flag: print_stateVehicle(StateVehicle)
        targetPoint = comp_rel_target(targetPoint,StateVehicle)#update the relative to the vehicle position of the target 
        cnt+=1
        if cnt>100:
            print("error, cannot compute distance_from_target_after_zeroing")
            break
    if print_flag: print("******end compute zeroing**********")
    
    return -targetPoint.rel_pos[0]


def comp_local_steer(nets,StateVehicle,targetPoint,trainHP,stop_flag):#return steer such that applying it the first step and then zeroing the steering results at the minimum distance from the target.
    #algorithm:
    #check dis at max steering
    #if stays at the same side as at the beginning - return max steering
    #else - find a dis from the same side, assume the zero steering is enongh
    #while dis < tolerance:
    #compute new steering according to the gradient and get the distance
    #chosse 

    tolerance = 0.05
    acc = 1.0#0
    init_dir = comp_steer_direction(targetPoint)
    #first guess the maximum dsteer - if distance is positive (cannot reach the point at none steering) return. 
    steer_max = get_dsteer_max(nets.SteerNet,StateVehicle.values,acc,init_dir,trainHP)
    dis_max = comp_distance_from_target_after_zeroing(nets,StateVehicle,targetPoint,acc,steer_max,trainHP,stop_flag)
    if print_flag: print("dis_max:",dis_max,"steer_max:",steer_max)
    if pause_by_user_flag: input("press to continue")
    if math.copysign(1,dis_max) == init_dir:#cannot reach the point with just the first point
        if print_flag: print("one step - dis_max:",dis_max,"steer_max:",steer_max)
        return steer_max
    else:#turn too sharp, hence can reach the point exactly
        steer_same =  0.0#steer_min
        dis_same = comp_distance_from_target_after_zeroing(nets,StateVehicle,targetPoint,acc,steer_same,trainHP,stop_flag)
        if math.copysign(1,dis_same) != init_dir:#cannot reach the point at all
            if print_flag: print("cannot reach target point, dis_min:",dis_same,"steer_min:",steer_same)
            return steer_same
        steer_not_same = steer_max
        dis_not_same = dis_max

        dis = dis_same #start with steer_min because it was not checked jet
        steer = steer_same#for the case that not going into the while loop
        dist_to_target = math.sqrt(targetPoint.rel_pos[0]**2+targetPoint.rel_pos[1]**2)
        cnt = 0
        while abs(dis) > max(dist_to_target*tolerance,tolerance) and not stop_flag:
            #print("while -comp_local_steer. stop_flag:",stop_flag)
            tmp = (dis_not_same - dis_same)
            if abs(tmp) < 1e-8:
                print("tmp<0")
                break
            else:
                steer = (steer_not_same - steer_same)/tmp*(-dis_same) + steer_same
                
                steer =np.clip(steer,-0.7,0.7).item()
                dis = comp_distance_from_target_after_zeroing(nets,StateVehicle,targetPoint,acc,steer,trainHP,stop_flag)
                if print_flag: print("compute new steer:\n steer_not_same:",steer_not_same,"dis_not_same:",dis_not_same,"steer_same:",
                      steer_same,"dis_same:",dis_same,"steer:",steer,"dis:",dis)

                if pause_by_user_flag: input("press to continue")

                if math.copysign(1,dis) == init_dir:
                    steer_same = steer
                    dis_same = dis
                else:
                    steer_not_same = steer
                    dis_not_same = dis
                dist_to_target = math.sqrt(targetPoint.rel_pos[0]**2+targetPoint.rel_pos[1]**2)

                cnt+=1
                if cnt>20:
                    print("error, cannot compute local steering")
                    break

        return steer


def step(stateVehicle,acc,steer,TransNet,trainHP):#get a state and actions, return next state
    steer = np.clip(steer,-0.7,0.7)
    x = [stateVehicle.values+[acc,steer]]
    y = TransNet.predict(np.array(x))[0]
    nextState = agent.State()
    delta_values = y[:len(trainHP.vehicle_ind_data)].tolist()
    nextState.Vehicle.values = [stateVehicle.values[i]+delta_values[i] for i in range(len(delta_values))]
    nextState.Vehicle.rel_pos = y[len(trainHP.vehicle_ind_data):len(trainHP.vehicle_ind_data)+2]
    nextState.Vehicle.rel_ang = y[len(trainHP.vehicle_ind_data)+2:]
    nextState.Vehicle.abs_pos,nextState.Vehicle.abs_ang = predict_lib.comp_abs_pos_ang(nextState.Vehicle.rel_pos,nextState.Vehicle.rel_ang,stateVehicle.abs_pos,stateVehicle.abs_ang)
    return nextState.Vehicle

def comp_steer(nets,StateVehicle,targetPoint,acc_flag,trainHP,stop_flag):
    line = None
    first_flag = True
    StateVehicle_vec = [copy.deepcopy(StateVehicle)]
    targetPoint_vec = [copy.deepcopy(targetPoint)]
    cnt = 0
    while ((targetPoint.rel_pos[1] > 0 and StateVehicle.values[0]>targetPoint.vel) or first_flag) and not stop_flag:
        #print("----------------------compute new step--------------------------. stop_flag:",stop_flag)
        if print_flag: print("----------------------compute new step--------------------------")
        if print_flag: print("current state:")
        if print_flag: print_stateVehicle(StateVehicle)
        if pause_by_user_flag: input("press to continue")
        

        
        steer = comp_local_steer(nets,StateVehicle,targetPoint,trainHP,stop_flag)#compute the steering independly of the acceleration
        if print_flag: print("computed steer:",steer)
        acc = 1.0 if acc_flag else -1.0 #tmp, in general case, from the AccNet
        if first_flag: first_steer,first_acc = steer,acc

        StateVehicle = step(StateVehicle,acc,steer,nets.TransNet,trainHP)
        targetPoint = comp_rel_target(targetPoint,StateVehicle)#update the relative to the vehicle position of the target
        StateVehicle_vec.append(copy.deepcopy(StateVehicle))
        targetPoint_vec.append(copy.deepcopy(targetPoint))

        acc_flag = False

        if plot_action_comp_flag: line = draw_state(StateVehicle,line = line)

        if print_flag: print_stateVehicle(StateVehicle)
         

        first_flag = False

        failed_flag = True if targetPoint.rel_pos[1] < 0 and StateVehicle.values[0]>targetPoint.vel else False # the vehicle will pass the target with too high velocity
        if cnt>100:
            print("error, cannot compute steering")
            break
    return failed_flag,first_acc,first_steer,StateVehicle_vec,targetPoint_vec

def comp_action(nets,state,trainHP,targetPoint,stop_flag):#compute the actions (and path) given the target point(s)
    #rel_pos of the vehicle - relative to last position (at last time step)
    #rel_pos of targetPoint - relative to the vehicle
    #abs_pos - relative to the initial pos, a new one at every time step.
    

    StateVehicle = state.Vehicle

    #vehicle_states = []
    if print_flag: print_stateVehicle(StateVehicle)

    #while dist_to_target < last_dist_to_target:#not always the right condition
    if plot_states_flag: draw_state(StateVehicle)
        
    #check if it is possible to accelerate
    acc_flag = True
    failed_flag,acc,steer,StateVehicle_vec,targetPoint_vec = comp_steer(nets,StateVehicle,targetPoint,acc_flag,trainHP,stop_flag)
    if failed_flag:
        acc_flag = False
        failed_flag,acc,steer,StateVehicle_vec,targetPoint_vec = comp_steer(nets,StateVehicle,targetPoint,acc_flag,trainHP,stop_flag)
    #print("acc:",acc,"steer:",steer)

    #input("press")
    #acc = comp_max_acc(nets.accNet,state.Vehicle.values,dsteer)
   
    #plt.ioff()
    #plt.show()
    return acc, steer,StateVehicle_vec,targetPoint_vec

def comp_action_from_next_step(nets,state,trainHP,targetPoint,acc,steer,stop_flag = False):
    
    state.Vehicle = step(state.Vehicle,acc,steer,nets.TransNet,trainHP)
    targetPoint = comp_rel_target(targetPoint,state.Vehicle)#update the relative to the vehicle position of the target
    acc,steer,StateVehicle_vec,targetPoint_vec = comp_action(nets,state,trainHP,targetPoint,stop_flag)
    return acc,steer,StateVehicle_vec,targetPoint_vec