from planner import Planner
import numpy as np
#from library import *  #temp
import library as lib
from classes import Path
import copy
import random
import time


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
    return points

def get_state(pl = None,local_path = None,points = 1,distance_between = 1):
    #velocity limit state:
    #velocity_limits = choose_points(local_path,points,distance_between)
    ## print("vel limit: ", velocity_limit)
    #vel = max(pl.simulator.vehicle.velocity,0)
    #state = [vel] +  velocity_limits
    #state = lib.normalize(state,0,30)

    #path state:



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
        reward = -10.*(velocity - velocity_limit)

    #if state[0] < 0: 
    #    reward = velocity_limit + state[0]
    #    if reward < 0.01:
    #        reward = - 2
    #else: 
    #    reward = -10*state[0]

    return reward

class ActionSpace:
    shape = []
    high = []
class ObservationSpace:
    shape = []

class OptimalVelocityPlanner:
    def __init__(self,dataManager):
        self.dataManager = dataManager
        self.max_episode_steps = 50
        self.feature_points = 60#30
        self.distance_between_points = 0.5 #meter 1.0
        self.features_num = 1 + self.feature_points #vehicle state points on the path (distance)
        self.path_lenght = self.feature_points*self.distance_between_points
        self.step_time = 0.2
        self.action_space_n = 1
        self.max_deviation = 3 # [m] if more then maximum - end episode 
        self.visualized_points = 300 #how many points show on the map and lenght of local path
        self.max_pitch = 0.3
        self.max_roll = 0.3
        self.acc = 1.5 # [m/s^2]  need to be more then maximum acceleration in real
        self. max_episode_length = 500
        self.action_space = ActionSpace()
        self.action_space.shape = [self.action_space_n]
        self.action_space.high = [1.0]#[self.acc*self.step_time] #temp, torque in Nm
        self.observation_space = ObservationSpace()
        self.observation_space.shape = [self.features_num]
        ####
        self.path_name = "paths\\path.txt"     #long random path: path3.txt  #long straight path: straight_path.txt
        self.path_source = "create_random"
        self.opened = False
        self.pl = Planner()
        if self.pl.connected:
            self.opened = True

        return
    
    def reset(self):
        self.episode_steps = 0
        self.last_time = [0]
        self.pl.restart()#not sure
        self.pl.simulator.get_vehicle_data()#read data after time step from last action
        if self.path_source == "saved_random" or self.path_source == "create_random":
            path_num = self.pl.load_path(self.path_lenght, self.path_name,self.path_source)
            if path_num == -1:#if cannot load path
                return "error"
                
        self.pl.new_episode()#compute path in current vehicle position
        #first state:
        local_path = self.pl.get_local_path()#num_of_points = self.visualized_points
        state = get_state(self.pl,local_path,self.feature_points,self.distance_between_points)
        return state

    def step(self, action):

        action = action[0]
        self.episode_steps+=1
        #pl.torque_command(a[0],max = action_space.high)
        #a = choose_action(action_space,Q[0],epsilon = HP.epsilon)
        #pl.delta_velocity_command(action_space[a])#update velocity (and steering) and send to simulator. index - index on global path (pl.desired_path)        
        self.pl.delta_velocity_command(action,self.acc)#
        print(action)
        #wait for step time:
        while (not lib.step_now(self.last_time,self.step_time)):# and stop != [True]: #wait for the next step (after step_time)
            time.sleep(0.00001)

        #get next state:
        self.pl.simulator.get_vehicle_data()#read data after time step from last action
        local_path = self.pl.get_local_path(send_path = False,num_of_points = self.visualized_points)#num_of_points = visualized_points
        next_state = get_state(self.pl,local_path,self.feature_points,self.distance_between_points)

        #get reward:
        #reward = get_reward(local_path.velocity_limit[0],self.pl.simulator.vehicle.velocity)
        reward = self.pl.simulator.vehicle.velocity / 30
        self.dataManager.update_real_path(pl = self.pl,velocity_limit = local_path.velocity_limit[0])
        self.dataManager.save_additional_data(features = lib.denormalize(next_state,0,30))#pl,features = denormalize(state,0,30),action = a
        mode = self.pl.check_end(self.path_lenght, deviation = lib.dist(local_path.position[0][0],local_path.position[0][1],0,0))#check if end of the episode 
        if self.episode_steps > self.max_episode_steps:
            mode = 'max_steps'
        if self.pl.simulator.vehicle.velocity > local_path.velocity_limit[0]:
            mode = 'cross'
        if mode != 'ok':
            if mode != 'path end' and mode != 'max_steps':
                reward = -5
            
            done = True
            if mode != 'kipp':
                self.pl.stop_vehicle()
            if  mode == 'kipp': #(i % HP.reset_every == 0 and i > 0) or
                #self.pl.stop_vehicle()
                self.pl.simulator.reset_position()
                self.pl.stop_vehicle()
        else:
            done = False
        print("reward", reward, "velocity: ", self.pl.simulator.vehicle.velocity, "mode:", mode)
        info = mode

        return next_state, reward, done, info
    def seed(self,seed_int):
        return#TODO set seed in unity
    def render(self):
        return#TODO set unity with / without graphics
    def close(self):
        #end all:
        print("del")
        self.pl.stop_vehicle()
        self.pl.end()
        


        