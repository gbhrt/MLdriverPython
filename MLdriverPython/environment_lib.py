import library as lib
import numpy as np
import math
import classes
import copy

def choose_velocity_limit_points(local_path,number,distance_between_points):#return points from the local path, choose a point every "skip" points
    index = 0
    last_index = 0
    points = []
    points.append(local_path.analytic_velocity_limit[index])
    for _ in range(number-1):
        while local_path.distance[index] - local_path.distance[last_index] < distance_between_points: #search the next point 
            if index >= len(local_path.distance)-1:#at the end of the path - break, cause doublication of the last velocity limit
                break
            index += 1
        points.append(local_path.analytic_velocity_limit[index])
        last_index = index
    return points
def choose_curvature_points(local_path,number,distance_between_points):#return points from the local path, choose a point every "skip" points
    index = 0
    last_index = 0
    points = []
    points.append(local_path.curvature[index])
    for _ in range(number-1):
        while local_path.distance[index] - local_path.distance[last_index] < distance_between_points: #search the next point 
            if index >= len(local_path.distance)-1:#at the end of the path - break, cause doublication of the last velocity limit
                break
            index += 1
        points.append(local_path.curvature[index])
        last_index = index
    return points
def choose_position_points(local_path,number,distance_between_points):#return points from the local path, choose a point every "skip" points
    index = 0
    last_index = 0
    end_flag = False
    points = []
    points.append(local_path.position[index][0])
    points.append(local_path.position[index][1])
    while len(points) < number*2 and end_flag == False:
        while local_path.distance[index] - local_path.distance[last_index] < distance_between_points: #search the next point 
            if index >= len(local_path.distance)-1:#
                end_flag = True
                break
            
            index += 1
        points.append(local_path.position[index][0])
        points.append(local_path.position[index][1])
        last_index = index
    
    ang = local_path.angle[-1][1]
    while len(points) < number * 2:
        #print(points)
        x = points[-2]
        y = points[-1]
        points.append(x + distance_between_points* math.sin(ang))#x
        points.append(y + distance_between_points* math.cos(ang))#y

    return points

def get_ddpg_target_state(pl = None,local_path = None,num_points = 1,distance_between = 1.0,max_velocity = 30.0,max_curvature = 0.12):
    #points = choose_position_points(local_path,num_points,distance_between)
    #max_lenght = distance_between*num_points
    #points = [pnt/max_lenght for pnt in points]
    

    #index = lib.select_target_index(local_path, velocity = pl.simulator.vehicle.velocity[1])
    index = 100
    points = [local_path.position[index][0],local_path.position[index][1]]

    vel = max(pl.simulator.vehicle.velocity[1]/max_velocity,0)
    state = [vel] +  points + [local_path.analytic_velocity[index]]
    return state

def get_ddpg_state(pl = None,local_path = None,num_points = 1,distance_between = 1.0,max_velocity = 30.0,max_curvature = 0.12):
    #velocity limit state:
    #points = choose_points(local_path,points,distance_between)

    #path state:
    points = choose_position_points(local_path,num_points,distance_between)
    max_lenght = distance_between*num_points
    points = [pnt/max_lenght for pnt in points]

    #curvature state:
    #points = choose_curvature_points(local_path,num_points,distance_between)
    #points = [pnt/max_curvature for pnt in points]

    
    
    vel = max(pl.simulator.vehicle.velocity[1]/max_velocity,0)
    state = [vel] +  points
    #state = lib.normalize(state,0,30)




    #path_ang = local_path.angle[0][1]
    #state.append(path_ang)
    #point1 = math.copysign(dist(local_path.position[0][0],local_path.position[0][1],0,0),local_path.position[0][0])#distance from path
    #state.append(point1)

    #for i in range (points):
    #    state.append(local_path.position[i][0])#x
    #    state.append(local_path.position[i][1])#y
    #state.append(local_path.position[target_index][0])#x
    #state.append(local_path.position[target_index][1])#y

    return state

def get_model_based_state(pl,last_abs_pos,last_abs_ang,local_path):
    rel_pos = lib.to_local(pl.simulator.vehicle.position,last_abs_pos,last_abs_ang[1])#relative position
    
    if rel_pos[0] >10:#temp error
        print ("error in get state__________________________________________________________________")
        print("pos:",pl.simulator.vehicle.position)

    rel_ang = pl.simulator.vehicle.angle[1] - last_abs_ang[1]#relative angle
    if rel_ang  > math.pi: rel_ang  -= 2*math.pi
    if rel_ang  < -math.pi: rel_ang  += 2*math.pi
    #vel = pl.simulator.vehicle.velocity[1]
    steer = pl.simulator.vehicle.steering
    roll = pl.simulator.vehicle.angle[2]
    #state = {'rel_pos':rel_pos,
    #         'rel_ang':rel_ang,
    #         'vel':vel,
    #         'steer':steer,
    #         'roll':roll,
    #         'path':local_path
    #         }

    state = {'rel_pos_x':rel_pos[0],
             'rel_pos_y':rel_pos[1],
             'rel_ang':rel_ang,
             'vel_x':pl.simulator.vehicle.velocity[0],
             'vel_y':pl.simulator.vehicle.velocity[1],
             'vel_z':pl.simulator.vehicle.velocity[2],
             'angular_vel_x':pl.simulator.vehicle.angular_velocity[0],
             'angular_vel_y':pl.simulator.vehicle.angular_velocity[1],
             'angular_vel_z':pl.simulator.vehicle.angular_velocity[2],
             'acc_x':pl.simulator.vehicle.acceleration[0],
             'acc_y':pl.simulator.vehicle.acceleration[1],
             'acc_z':pl.simulator.vehicle.acceleration[2],
             'angular_acc_x':pl.simulator.vehicle.angular_acceleration[0],
             'angular_acc_y':pl.simulator.vehicle.angular_acceleration[1],
             'angular_acc_z':pl.simulator.vehicle.angular_acceleration[2],
             'wheel_n_vel':[pl.simulator.vehicle.wheels[0].vel_n,pl.simulator.vehicle.wheels[2].vel_n],#[wheel.vel_n for wheel in pl.simulator.vehicle.wheels],
             'steer':steer,
             'roll':roll,
             'path':local_path
             }

    last_abs_pos[0] = pl.simulator.vehicle.position[0]
    last_abs_pos[1] = pl.simulator.vehicle.position[1]
    last_abs_pos[2] = pl.simulator.vehicle.position[2]
    last_abs_ang[1] = pl.simulator.vehicle.angle[1]
    return state

#def get_DDPG_target_reward(state,velocity,max_vel,roll,mode,deviation,lower_bound = 0.0):
#    if mode == 'kipp'or mode == 'deviate':
#        reward = -1.0
#    else:
#        reward =  -abs(velocity)/max_vel - abs(roll)#0.02*
#    return reward

def get_SDDPG_reward_stabilize(velocity,max_vel,roll,mode,deviation,lower_bound = 0.0):
    if mode == 'kipp'or mode == 'deviate':
        reward = -1.0
    else:
        reward =  0.01*(-abs(velocity)/max_vel - abs(roll))#0.02*
    return reward
def get_SDDPG_reward(ang,velocity,max_vel,roll,mode,deviation,lower_bound = 0.0,steer = None):
    #print("progress:",progress,"deviation:",deviation)
    if steer is not None:
        violation_flag = get_direct_stability(velocity,steer)
    else:
        violation_flag = False
    if mode == 'kipp'or mode == 'deviate':
        reward = -1.0
    elif velocity <= lower_bound:
        reward = -0.2
    elif violation_flag:
        print("violation_flag")
        reward = -0.5
    elif abs(deviation) > 4.0:
        print("deviation > 4.0")
        reward = -0.1

    # elif  abs(roll) > 0.1:
    #     reward = -0.5
    else:
        reward = velocity/max_vel# *np.cos(ang)#progress/(max_vel*0.2/0.05)
    #print("reward:",reward)
    #else:
    #    #reward =  - abs(deviation*0.03)+0.02*velocity/max_vel
    #    reward =  - abs(deviation*0.03)+0.02*progress/(max_vel*0.2/0.05)
    return reward
def comp_LTR(vel,steer):
    if abs(steer) < 0.001:
        radius = 1000
    else:
        radius = math.sqrt((3.6*0.5)**2+(3.6/math.tan(steer))**2)
    if radius <0.1:
        print("error radius too small")
        ac = 100
    else:
        ac = vel**2/radius

    return ac/(9.81*2.08*0.5/1.0)#maximal cetripetal force

def get_direct_stability(velocity,steer):
    LTR = comp_LTR(velocity,steer)
    print("LTR:",LTR)
    if LTR > 1.0:
        return True
    else:
        return False
def get_reward(velocity,max_vel,mode,lower_bound = 0.0,analytic_velocity = None,roll = None,max_alowed_roll = None,max_roll =None,steer = None): 
    if analytic_velocity is not None:
        #reward = -abs((velocity - analytic_velocity)/max_vel)
        if velocity < analytic_velocity:
            reward = (velocity - analytic_velocity)/max_vel #negative reward
        else:
            reward = (velocity - analytic_velocity)/max_vel #positive reward
        print("velocity:",velocity,"analytic_velocity:",analytic_velocity,"reward:",reward)
    else:
        #reward = 0.2*velocity/max_vel 
         reward = 0.02*velocity/max_vel 
        #if velocity <= 0.0:
    if velocity <= lower_bound:
        #reward = -0.2
        reward = -0.2
    #if roll is not None:
    #    if roll > max_alowed_roll:
    #        reward = 0.02*velocity/max_vel - roll/max_roll
    if steer is not None:
        violation_flag = get_direct_stability(velocity,steer)
        reward = -0.5

    if mode == 'kipp'or mode == 'deviate':
        reward = -1.0
    #elif mode == 'path_end':#problem - agent dont know that he is at the end
    #    reward = 10

    
    #if state[1] < 0.01: # finished the path
    #    reward =(max_velocity - state[0])
    #else:
    #    reward =  0#state[0]*0.01


    #velocity_limit = state[1]
    #acc = state[0] - last_state[0]#acceleration
    #if state[0] < velocity_limit:
    #    if last_state[0] > velocity_limit:
    #        reward = -acc 
    #    else:
    #        reward = acc 
    #else:

    #    reward = -acc
    #acc = state[0] - last_state[0]#acceleration
    #if state[0] < 0:
    #    if last_state[0] > velocity_limit:
    #        reward = -acc 
    #    else:
    #        reward = acc 
    #else:
    #    reward = -acc
    #if velocity < velocity_limit: 
    #    reward = velocity
    #    #if velocity < 1e-5:
    #    #    reward = - 2
    #else: 
    #    reward = -10.*(velocity - velocity_limit)

    #if state[0] < 0: 
    #    reward = velocity_limit + state[0]
    #    if reward < 0.01:
    #        reward = - 2
    #else: 
    #    reward = -10*state[0]

    return reward
