import library as lib
import numpy as np
import agent
import matplotlib.pyplot as plt
import copy
import math 

import predict_lib

#debug settings:
print_flag = True
plot_local_steer_comp_flag = True
plot_action_comp_flag = True
plot_check_reachability_flag = True
plot_states_flag = True

pause_by_user_flag = True

#fig,[ax_abs,ax_rel] = plt.subplots(2)
#ax_rel.axis('equal')

#if plot_states_flag or plot_action_comp_flag or plot_local_steer_comp_flag:
    #fig,ax_abs = plt.subplots(1)
    #ax_abs.axis('equal')
    #plt.ion()





def print_stateVehicle(StateVehicle):
    print("abs_pos:",StateVehicle.abs_pos,"abs_ang:",StateVehicle.abs_ang,"\n values:",StateVehicle.values)

def plot_state(StateVehicle,ax,line = None):
    if line is None:
        line, = ax.plot([StateVehicle.abs_pos[0]],[StateVehicle.abs_pos[1]],'.')

    else:
        ax.plot([StateVehicle.abs_pos[0]],[StateVehicle.abs_pos[1]],'.',color = line.get_color())
    ax.plot([StateVehicle.abs_pos[0],StateVehicle.abs_pos[0]+math.sin(StateVehicle.abs_ang)],[StateVehicle.abs_pos[1],StateVehicle.abs_pos[1]+math.cos(StateVehicle.abs_ang)],color = line.get_color())
    
def draw_state(StateVehicle,ax,line = None):
    plot_state(StateVehicle,ax,line = line)
    plt.draw()
    plt.pause(0.0001)
    return line

def plot_target(targetPoint,ax):
    ax.plot([targetPoint.abs_pos[0]],[targetPoint.abs_pos[1]],'x')

def draw_target(targetPoint,ax):
    plot_target(targetPoint,ax)
    plt.draw()
    plt.pause(0.0001)

def plot_state_vec(StateVehicle_vec,ax):
    for StateVehicle in StateVehicle_vec:
        plot_state(StateVehicle,ax)



def steer_policy(state_Vehicle,state_env,trainHP,SteerNet = None):
    #acc = 1.0
    #steer_max = get_dsteer_max(SteerNet,state.Vehicle.values,acc,1,trainHP)
    #steer_min = get_dsteer_max(SteerNet,state.Vehicle.values,acc,-1,trainHP)
    #print("steer:",steer_min,steer_max)
    #steer_max = np.clip(0.7-state.Vehicle.values[0]*0.1,0,0.7)
    #steer_min = np.clip(-(0.7-state.Vehicle.values[0]*0.1),-0.7,0)
    #np.clip(steer,steer_min,steer_max).item()
    steer = lib.comp_steer_general(state_env[0],state_env[1],state_Vehicle.abs_pos,state_Vehicle.abs_ang,state_Vehicle.values[0])
    return np.clip(steer,-0.7,0.7).item()

#def emergency_steer_policy(state):
#    return 0.0
def emergency_steer_policy(state_Vehicle,state_env,trainHP,SteerNet = None):
    if trainHP.emergency_steering_type == 1:
        steer = 0
    elif trainHP.emergency_steering_type == 2:
        steer = 0.5*lib.comp_steer_general(state_env[0],state_env[1],state_Vehicle.abs_pos,state_Vehicle.abs_ang,state_Vehicle.values[0])
    elif trainHP.emergency_steering_type == 3:
        steer = 0
        acc = -1.0
        steer_max = get_dsteer_max(SteerNet,state_Vehicle.values,acc,1,trainHP)
        steer_min = get_dsteer_max(SteerNet,state_Vehicle.values,acc,-1,trainHP)
        steer = np.clip(steer,steer_min,steer_max).item()
    elif trainHP.emergency_steering_type == 4:
        #print("continue same steering")
        steer = state_Vehicle.values[1]#continue same state
    elif trainHP.emergency_steering_type == 5:#propotional to roll angle
        #print("stabilize by P")
        k = 3.5
        steer =np.clip(k*state_Vehicle.values[2],-0.7,0.7).item()

    else:
        print("error - emergency_steering_type not exist")
    return steer

def acc_policy():
    return -1.0
def emergency_acc_policy():
    return -1.0
def clip_steering(state,acc,steer,SteerNet,trainHP):
    steer_max = get_dsteer_max(SteerNet,state.Vehicle.values,acc,1,trainHP)
    steer_min = get_dsteer_max(SteerNet,state.Vehicle.values,acc,-1,trainHP)
    cliped = False
    if steer > steer_max:
        steer = steer_max
        cliped = True
    elif steer < steer_min:
        steer = steer_min
        cliped = True
    return cliped

def check_stability(state_Vehicle,state_env,roll_var = 0.0,max_plan_roll = None,max_plan_deviation = None):
    dev_flag,roll_flag = 0,0
    path = state_env[0]
    index = state_env[1]
    roll = state_Vehicle.values[2]
    #print("roll",roll,"roll var",roll_var)
    dev_from_path = lib.dist(path.position[index][0],path.position[index][1],state_Vehicle.abs_pos[0],state_Vehicle.abs_pos[1])#absolute deviation from the path
    if abs(roll)+roll_var > max_plan_roll: #check the current roll 
        roll_flag = math.copysign(1,roll)
    if dev_from_path > max_plan_deviation:
        dev_flag = 1
        #max_plan_deviation = 10
    return roll_flag,dev_flag

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




def get_dsteer_max(SteerNet,StateVehicle,acc, direction,trainHP):#get maximum change in steering in a given state
    current_roll = StateVehicle[2]
    des_roll = -trainHP.plan_roll*direction
    #droll = des_roll - current_roll
    #print("droll:",des_roll)
    steer_max = SteerNet.predict(np.array([StateVehicle+[acc,des_roll]]))[0][0]
    return np.clip(steer_max,-0.7,0.7).item()

def comp_steer_direction(targetPoint):
    return math.copysign(1.0, -targetPoint.rel_pos[0])

def comp_max_steer_radius(TransNet,dir,trainHP):
    StateVehicle = agent.State().Vehicle
    steer = 0.7*dir
    StateVehicle.values = [0.0,steer,0.0]
    StateVehicle.abs_ang = 0.0
    StateVehicle.abs_pos = [0.0,0.0]
    acc = 1.0
    StateVehicle = step(StateVehicle,acc,steer,TransNet,trainHP)
    dang = abs(StateVehicle.rel_ang)
    d = math.sqrt(StateVehicle.rel_pos[0]**2+StateVehicle.rel_pos[1]**2)#
    tmp = math.sin(dang)
    if abs(tmp)>1e-6:
        r = d*math.sin((math.pi-dang)/2)/tmp
    else:
        r = d
    return abs(r)

def check_quasistatic_reachability(TransNet,targetPoint,trainHP):
    r = comp_max_steer_radius(TransNet,comp_steer_direction(targetPoint),trainHP)
    print("radius =",r)
    dis_to_target = math.sqrt(targetPoint.rel_pos[0]**2+targetPoint.rel_pos[1]**2)
    return dis_to_target > r

def check_reachability(nets,StateVehicle,targetPoint,trainHP,stop_flag,ax = None):
    #try to turn as much as possible until stop, then continue quasistatic turn (max curvature of the vehicle at (near to) zero velocity)
    if print_flag: print("check_reachability:")
    if plot_check_reachability_flag: line = draw_state(StateVehicle,ax)
    targetPoint = comp_rel_target(targetPoint,StateVehicle)
    init_dir = comp_steer_direction(targetPoint)
    
    acc = -1.0#0.0
    cnt = 0
    while StateVehicle.values[0] > 0 and not stop_flag:#target in front of the vehicle
        dir = comp_steer_direction(targetPoint)
        if dir != init_dir:
            return True
        steer = get_dsteer_max(nets.SteerNet,StateVehicle.values,acc,dir,trainHP)
        StateVehicle = step(StateVehicle,acc,steer,nets.TransNet,trainHP)
        if plot_check_reachability_flag: draw_state(StateVehicle,ax,line = line)
        if print_flag: print_stateVehicle(StateVehicle)
        targetPoint = comp_rel_target(targetPoint,StateVehicle)#update the relative to the vehicle position of the target 
        cnt+=1
        if cnt>100:
            print("error, cannot compute check_reachability")
            break
        #if reached the target line before stopping
        if targetPoint.rel_pos[1]<0:
            if targetPoint.rel_pos[0]< trainHP.min_dis:
                return True
            else:
                 return False
        
    return check_quasistatic_reachability(nets.TransNet,targetPoint,trainHP)

def comp_distance_from_target_after_zeroing(nets,StateVehicle,targetPoint,trainHP,stop_flag,ax = None):#get state and actions, return the x of relative distance to target point
    targetPoint = comp_rel_target(targetPoint,StateVehicle)#update the relative to the vehicle position of the target 
    if print_flag: print("******compute zeroing**********")

    if plot_local_steer_comp_flag: line = draw_state(StateVehicle,ax)

    if print_flag: print_stateVehicle(StateVehicle)
    steer = 0.0
    acc = -1.0#0.0
    cnt = 0
    while abs(StateVehicle.values[1]) > 0.01 and targetPoint.rel_pos[1]>0 and StateVehicle.values[0] > 0 and not stop_flag:#steering > 0 and target in front of the vehicle
        #print("while - comp_distance_from_target_after_zeroing. stop_flag:",stop_flag)
        StateVehicle = step(StateVehicle,acc,steer,nets.TransNet,trainHP)
        if plot_local_steer_comp_flag: draw_state(StateVehicle,ax,line = line)

        if print_flag: print_stateVehicle(StateVehicle)
        targetPoint = comp_rel_target(targetPoint,StateVehicle)#update the relative to the vehicle position of the target 
        cnt+=1
        if cnt>100:
            print("error, cannot compute distance_from_target_after_zeroing")
            break
    if print_flag: print("******end compute zeroing**********")
    
    return -targetPoint.rel_pos[0]


def comp_local_steer(nets,StateVehicle,targetPoint,trainHP,stop_flag,ax = None):#return steer such that applying it the first step and then zeroing the steering results at the minimum distance from the target.
    #algorithm:
    #check dis at max steering
    #if stays at the same side as at the beginning - return max steering
    #else - find a dis from the same side, assume the zero steering is enongh
    #while dis < trainHP.target_tolerance:
    #compute new steering according to the gradient and get the distance
    #chosse 

    
    acc = 1.0#0
    init_dir = comp_steer_direction(targetPoint)
    #first guess the maximum dsteer - if distance is positive (cannot reach the point at none steering) return. 
    steer_max = get_dsteer_max(nets.SteerNet,StateVehicle.values,acc,init_dir,trainHP)
    StateVehicle = step(StateVehicle,acc,steer_max,nets.TransNet,trainHP)
    dis_max = comp_distance_from_target_after_zeroing(nets,StateVehicle,targetPoint,trainHP,stop_flag,ax)
    if print_flag: print("dis_max:",dis_max,"steer_max:",steer_max)
    if pause_by_user_flag: input("press to continue")
    if math.copysign(1,dis_max) == init_dir:#cannot reach the point with just the first point
        if print_flag: print("one step - dis_max:",dis_max,"steer_max:",steer_max)
        return  steer_max,False
    else:#turn too sharp, hence can reach the point exactly
        steer_same =  0.0#steer_min
        StateVehicle = step(StateVehicle,acc,steer_same,nets.TransNet,trainHP)
        dis_same = comp_distance_from_target_after_zeroing(nets,StateVehicle,targetPoint,trainHP,stop_flag,ax)
        if math.copysign(1,dis_same) != init_dir:#cannot reach the point at all
            if print_flag: print("cannot reach target point, dis_min:",dis_same,"steer_min:",steer_same)
            return steer_same,True
        steer_not_same = steer_max
        dis_not_same = dis_max

        dis = dis_same #start with steer_min because it was not checked jet
        steer = steer_same#for the case that not going into the while loop
        dist_to_target = math.sqrt(targetPoint.rel_pos[0]**2+targetPoint.rel_pos[1]**2)
        cnt = 0
        while abs(dis) > max(dist_to_target*trainHP.target_tolerance,trainHP.target_tolerance) and not stop_flag:
            #print("while -comp_local_steer. stop_flag:",stop_flag)
            tmp = (dis_not_same - dis_same)
            if abs(tmp) < 1e-8:
                print("tmp<0")
                break
            else:
                steer = (steer_not_same - steer_same)/tmp*(-dis_same) + steer_same
                
                steer =np.clip(steer,-0.7,0.7).item()
                StateVehicle = step(StateVehicle,acc,steer,nets.TransNet,trainHP)
                dis = comp_distance_from_target_after_zeroing(nets,StateVehicle,targetPoint,trainHP,stop_flag,ax)
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

        return steer,True


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

def emergency_step(stateVehicle,acc,steer,Direct):#get a state and actions, return next state
    steer = np.clip(steer,-0.7,0.7)
    next_vehicle_state,rel_pos,rel_ang = Direct.predict_one(stateVehicle.values,[acc,steer])
    nextState = agent.State()
    nextState.Vehicle.values = [stateVehicle.values[i]+next_vehicle_state[i] for i in range(len(next_vehicle_state))]
    nextState.Vehicle.rel_pos = rel_pos
    nextState.Vehicle.rel_ang = rel_ang
    #nextState.Vehicle.abs_pos,nextState.Vehicle.abs_ang = predict_lib.comp_abs_pos_ang(nextState.Vehicle.rel_pos,nextState.Vehicle.rel_ang,stateVehicle.abs_pos,stateVehicle.abs_ang)
    nextState.Vehicle.abs_pos,nextState.Vehicle.abs_ang = rel_pos,rel_ang
    return nextState.Vehicle


def stop_toward_target(nets,StateVehicle,targetPoint,acc_flag,trainHP,stop_flag,ax = None):
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
        

        
        steer,reach_flag = comp_local_steer(nets,StateVehicle,targetPoint,trainHP,stop_flag,ax)#compute the steering independly of the acceleration
        if print_flag: print("computed steer:",steer)
        acc = 1.0 if acc_flag else -1.0 #tmp, in general case, from the AccNet
        if first_flag: first_steer,first_acc = steer,acc

        StateVehicle = step(StateVehicle,acc,steer,nets.TransNet,trainHP)
        targetPoint = comp_rel_target(targetPoint,StateVehicle)#update the relative to the vehicle position of the target

        StateVehicle_vec.append(copy.deepcopy(StateVehicle))
        targetPoint_vec.append(copy.deepcopy(targetPoint))

        if pause_by_user_flag: input("press to continue")

        acc_flag = False

        if plot_action_comp_flag: line = draw_state(StateVehicle,ax,line = line)

        if print_flag: print_stateVehicle(StateVehicle)
         

        first_flag = False

        if cnt>100:
            print("error, cannot compute steering")
            break

    #end while

    if not reach_flag:
        reach_flag =  check_reachability(nets,StateVehicle,targetPoint,trainHP,stop_flag,ax)
        if not reach_flag:
            print("cannot reach target")
    if pause_by_user_flag: input("press to continue")

    failed_flag = True if (targetPoint.rel_pos[1] < 0 and StateVehicle.values[0]>targetPoint.vel) or not reach_flag else False # the vehicle will pass the target with too high velocity
    print("failed_flag:",failed_flag)
    return failed_flag,first_acc,first_steer,StateVehicle_vec,targetPoint_vec


def comp_abs_dsteer(nets,StateVehicle,targetPoint,trainHP,stop_flag,ax = None):
    steer,reach_flag = comp_local_steer(nets,StateVehicle,targetPoint,trainHP,stop_flag,ax)
    current_steer = StateVehicle.values[1]
    acc = -1.0#maybe 1.0
    StateVehicle = step(StateVehicle,acc,steer,nets.TransNet,trainHP)
    next_steer = StateVehicle.values[1]

    return abs(next_steer) - abs(current_steer)

def comp_action(nets,StateVehicle,trainHP,targetPoint,stop_flag,ax = None):#compute the actions (and path) given the target point(s)
    #rel_pos of the vehicle - relative to last position (at last time step)
    #rel_pos of targetPoint - relative to the vehicle
    #abs_pos - relative to the initial pos, a new one at every time step.
    

    #vehicle_states = []
    if print_flag: print_stateVehicle(StateVehicle)

    #while dist_to_target < last_dist_to_target:#not always the right condition
    if plot_states_flag: draw_state(StateVehicle,ax)
        
    abs_dsteer = comp_abs_dsteer(nets,StateVehicle,targetPoint,trainHP,stop_flag,ax)
    print("abs_dsteer:",abs_dsteer)
    if abs_dsteer > 0:
        acc_flag = False
    else:
        #check if it is possible to accelerate at the first step and not exceed the target velocity:
        acc_flag = True
        failed_flag,acc,steer,StateVehicle_vec,targetPoint_vec = stop_toward_target(nets,StateVehicle,targetPoint,acc_flag,trainHP,stop_flag,ax)
        if failed_flag: acc_flag = False

    if not acc_flag:
        failed_flag,acc,steer,StateVehicle_vec,targetPoint_vec = stop_toward_target(nets,StateVehicle,targetPoint,acc_flag,trainHP,stop_flag,ax)
    #print("acc:",acc,"steer:",steer)


    #check if it is possible to accelerate
    #acc_flag = True
    #failed_flag,acc,steer,StateVehicle_vec,targetPoint_vec = stop_toward_target(nets,StateVehicle,targetPoint,acc_flag,trainHP,stop_flag,ax)
    #if failed_flag:
    #    acc_flag = False
    #    failed_flag,acc,steer,StateVehicle_vec,targetPoint_vec = stop_toward_target(nets,StateVehicle,targetPoint,acc_flag,trainHP,stop_flag,ax)
    #print("acc:",acc,"steer:",steer)

    #input("press")
    #acc = comp_max_acc(nets.accNet,state.Vehicle.values,dsteer)
   
    #plt.ioff()
    #plt.show()
    return acc, steer,StateVehicle_vec,targetPoint_vec

def comp_action_from_next_step(nets,state,trainHP,targetPoint,acc,steer,stop_flag = False,ax = None):
    
    state.Vehicle = step(state.Vehicle,acc,steer,nets.TransNet,trainHP)
    targetPoint = comp_rel_target(targetPoint,state.Vehicle)#update the relative to the vehicle position of the target
    acc,steer,StateVehicle_vec,targetPoint_vec = comp_action(nets,state.Vehicle,trainHP,targetPoint,stop_flag,ax)
    return acc,steer,StateVehicle_vec,targetPoint_vec