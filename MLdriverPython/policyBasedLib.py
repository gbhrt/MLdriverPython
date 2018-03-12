import copy
import random
import numpy as np
import json

def choose_points(local_path,number,distance_between_points):#return points from the local path, choose a point every "skip" points
    index = 0
    last_index = 0
    points = []
    points.append(local_path.velocity_limit[index])
    for _ in range(number-1):
        while local_path.distance[index] - local_path.distance[last_index] < distance_between_points: #search the next point 
            if index >= len(local_path.distance)-1:#at the end of the path - break, cause doublication of the last velocity limit
                break
            index += 1
        points.append(local_path.velocity_limit[index])
        last_index = index


    #n = 0
    #points = []
    #while n < len(local_path.position):
    #    points.append(local_path.position[n][0])
    #    points.append(local_path.position[n][1])
    #    if len(points) == number * 2:
    #        break
    #    n+=skip
    #dis = 2
    #ang = local_path.angle[-1][1]
    #while len(points) < number * 2:
    #    points.append(local_path.position[-1][0] + dis* math.sin(ang))
    #    points.append(local_path.position[-1][1] + dis* math.cos(ang))
    return points

 

def get_state(pl = None,local_path = None,points = 1,distance_between = 1):
    state = []
    velocity_limits = choose_points(local_path,points,distance_between)
    # print("vel limit: ", velocity_limit)
    vel = pl.simulator.vehicle.velocity
    for i in range(points):
        state.append(vel - velocity_limits[i])
        
    #i = find_low_vel(local_path)
    #dis = local_path.distance[i]
    #state.append(dis)

    #state += choose_points(local_path,points,30)

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

def get_reward(last_state,state,velocity_limit):   
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
    if state[0] < 0: 
        reward = velocity_limit + state[0]
        if reward < 0.01:
            reward = - 2
    else: 
        reward = -10*state[0]

    return reward

def choose_action(action_space,Pi,steps = None):
    #epsilon = 0.0
    #if random.random() < epsilon:
    #    a = random.randint(0,len(action_space) - 1)#random.randint(0,(len(action_space.data) - 1))
    #    print("random a: ",a)
    #else:
    #    a = np.argmax(Pi)
    #    print("best a: ",a)
    
    #action = np.random.choice(list_of_candidates, number_of_items_to_pick, p=probability_distribution)

    #choose a random action for the next random steps:
    #if steps[0] == 0:
    #    steps[0] = random.randint(1,10)
    #    steps[1] = random.randint(0,len(action_space) - 1)
    #    print("choose: steps: ",steps[0],"action: ",steps[1])
    #a = steps[1]
    #steps[0] -= 1
    #print("steps: ",steps[0])

    #choose action from propbilities of Pi:
    rand = random.random()
    prob = 0
    for i in range(len(action_space)):
        prob += Pi[i]
        if rand < prob:
            a =  i
            break

    #always the highest probability:
    #a = np.argmax(Pi)


    if a == 1:
        one_hot_a = [0,1]
    else:
        one_hot_a = [1,0]
    return a,one_hot_a

def comp_Pi(net):
    for v0 in range(-5,5):
        for v in range(-5,5):
            Pi = net.get_Pi([v0,v])
            print(Pi,'\t', end='')

def comp_value(net,max_vel):
    distance = 10
    for v in range(max_vel):
        for p in range(10):
            #Q = net.get_Q([v,p])
            #A = sum(Q) / float(len(Q))
            #print(Q,'\t', end='')
            #V = net.get_V([v,p])
            #print(V,'\t', end='')
            Pi = net.get_Pi([v,p])
            print(Pi,'\t', end='')

        print()

