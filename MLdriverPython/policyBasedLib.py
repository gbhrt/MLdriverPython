import copy
import random
import numpy as np
import json

class HyperParameters:
    def __init__(self):
        self.feature_points = 10 #
        self.distance_between_points = 1.0 #meter
        self.features_num = 1 + self.feature_points #vehicle state points on the path (distance)
        self.epsilon_start = 1.0
        self.epsilon = 0.1
        self.gamma = 0.99
        self.tau = 0.01 #how to update target network compared to Q network
        self.num_of_runs = 5000
        self.step_time = 0.2
        self.alpha_actor = 0.001# for Pi 1e-5 #learning rate
        self.alpha_critic = 0.001#for Q
        self.max_deviation = 3 # [m] if more then maximum - end episode 
        self.batch_size = 32
        self.replay_memory_size = 10000
        self.num_of_TD_steps = 15 #for TD(lambda)
        self.visualized_points = 300 #how many points show on the map - just visualy
        self.max_pitch = 0.3
        self.max_roll = 0.3
        self.acc = 1.5 # [m/s^2]  need to be more then maximum acceleration in real
        self.res = 1
        self.plot_flag = True
        self.restore_flag = True
        self.skip_run = False
        self.random_paths_flag = True
        self.reset_every = 3
        self.save_every = 25
        self.path_name = "paths\\path.txt"     #long random path: path3.txt  #long straight path: straight_path.txt
        self.save_name = "model17" #model6.ckpt - constant velocity limit - good. model7.ckpt - relative velocity.
        #model10.ckpt TD(5) dt = 0.2 alpha 0.001 model13.ckpt - 5 points 2.5 m 0.001 TD 15
        #model8.ckpt - offline trained after 5 episode - very clear
        self.restore_name = "model17" # model2.ckpt - MC estimation 
        self.run_data_file_name = 'running_record1'

class Replay:
    def __init__(self,replay_memory_size):
        self.memory_size = replay_memory_size
        self.memory = []
    def add(self,data):
        if len(self.memory) > self.memory_size:
            self.memory.pop(0)
        self.memory.append(data)
    def sample(self,batch_size):
        samples = random.sample(self.memory,np.clip(batch_size,0,len(self.memory)))
        return map(np.array, zip(*samples))
        


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

 
def normalize(val,min,max):
    nval = []
    for item in val:
        nval.append( (item - min)/(max - min))
    return nval
def denormalize(nval, min, max):
    val = []
    for item in nval:
        val.append( item * (max - min) + min)
    return val

def get_state(pl = None,local_path = None,points = 1,distance_between = 1):
    #state = []
    velocity_limits = choose_points(local_path,points,distance_between)
    # print("vel limit: ", velocity_limit)
    vel = max(pl.simulator.vehicle.velocity,0)
    state = [vel] +  velocity_limits
    #for i in range(points):
    #    state.append(vel - velocity_limits[i])
    state = normalize(state,0,30)
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

def get_reward(velocity_limit,velocity):   
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
    if velocity < velocity_limit: 
        reward = velocity
        #if velocity < 1e-5:
        #    reward = - 2
    else: 
        reward = -1.*(velocity - velocity_limit)

    #if state[0] < 0: 
    #    reward = velocity_limit + state[0]
    #    if reward < 0.01:
    #        reward = - 2
    #else: 
    #    reward = -10*state[0]

    return reward

def choose_action(action_space,Pi,steps = None,epsilon = 0.1):
    if random.random() < epsilon:
        a = random.randint(0,len(action_space) - 1)#random.randint(0,(len(action_space.data) - 1))
        print("random a: ",a)
    else:
        a = np.argmax(Pi)
        print("best a: ",a)
    
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
    #rand = random.random()
    #prob = 0
    #for i in range(len(action_space)):
    #    prob += Pi[i]
    #    if rand < prob:
    #        a =  i
    #        break

    #always the highest probability:
    #a = np.argmax(Pi)

    return a

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

