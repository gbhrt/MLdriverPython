from planner import Planner
import numpy as np
#from library import *  #temp
import library as lib
import classes
import copy
import random
import os
import time

from enviroment_lib import *
class ActionSpace:
    shape = []
    high = []
class ObservationSpace:
    shape = []


class OptimalVelocityPlannerData:
    def __init__(self):
        self.analytic_feature_flag = False
        self.end_indication_flag = False
        self.max_episode_steps = 100#200#
        self.feature_points = 25 # number of points on the path in the state +(1*self.end_indication_flag)
        self.feature_data_num = 1  + (1*self.analytic_feature_flag) # number of data in state (vehicle velocity, analytic data...)
        self.features_num = self.feature_data_num +  2*self.feature_points #vehicle state points on the path (distance)
        self.distance_between_points = 1.0 #meter 1.0
        self.path_lenght = 9000#self.feature_points*self.distance_between_points
        self.step_time = 0.2
        self.action_space_n = 1
        self.max_deviation = 3 # [m] if more then maximum - end episode 
        self.visualized_points = int(self.feature_points/0.05) + 10 #how many points show on the map and lenght of local path
        self.max_pitch = 0.3#0.3
        self.max_roll = 0.07#0.05#0.3
        self.acc = 1.38# 0-100 kmh in 20 sec. 1.5 # [m/s^2]  need to be more then maximum acceleration in real
        self.torque_reduce = 1.0 # 0.2
        self.max_episode_length = 500
        self.action_space = ActionSpace()
        self.action_space.shape = [self.action_space_n]
        self.action_space.high = [1.0]#self.acc*self.step_time] #, torque in Nm
        self.observation_space = ObservationSpace()
        self.observation_space.shape = [self.features_num]
        self.max_velocity = 30
        self.max_curvature = 0.12

        

class OptimalVelocityPlanner(OptimalVelocityPlannerData):
    def __init__(self,dataManager):
        super().__init__()
        self.dataManager = dataManager
       
        ####
        self.path_name = "paths//straight_path_limit_json.txt"     #long random path: path3.txt  #long straight path: straight_path.txt
        self.path_source = "create_random" #  "regular" #"create_random" #saved_random"
      
        self.reset_count = 0
        self.reset_every = 5
   
        self.pl = Planner("torque")
        self.opened = self.pl.connected

        self.path_seed = None

        return

        
             

    def reset(self,seed = None):
        
        if (self.reset_count % self.reset_every == 0 and self.reset_count > 0): 
            self.pl.simulator.reset_position()
            self.pl.stop_vehicle()
        self.episode_steps = 0
        self.last_time = [0]
        self.pl.restart()#not sure
        self.pl.simulator.get_vehicle_data()#read data after time step from last action
        #if self.path_source == "saved_random" or self.path_source == "create_random":
        if seed == None:
            seed = int.from_bytes(os.urandom(8), byteorder="big")
            self.dataManager.path_seed.append(seed)
   #     else:
            #if len(self.dataManager.path_seed) > path_num:
            #    seed = self.dataManager.path_seed[path_num]
            #else:
            #    print("error - path id not exist")

        path_error = self.pl.load_path(self.path_lenght, self.path_name,self.path_source,seed = seed)#,seed = 1236
        self.path_seed = seed
        if path_error == -1:#if cannot load path
            return "error"
        
        self.pl.new_episode()#compute path in current vehicle position
        #self.dataManager.update_planned_path(self.pl.in_vehicle_reference_path)
        #first state:
        local_path = self.pl.get_local_path()#num_of_points = self.visualized_points
        state = get_state(self.pl,local_path,self.feature_points,self.distance_between_points,self.max_velocity,self.max_curvature)
        if self.analytic_feature_flag:
            analytic_a = [self.comp_analytic_acceleration(state)]
            state = analytic_a + state
        #if self.end_indication_flag:
        #    dis_from_end = self.features_num*self.feature_points - self.pl.in_vehicle_reference_path.distance[self.pl.main_index]
            #self.dataManager.add(('curvature',(self.pl.desired_path.curvature)))
 
        self.reset_count+=1
        return state

    def command(self,action):# the algo assume that the action is in the last state, must be close as possible to the end of the
                                #last step command, hence the training is between command and state update (while waiting for step time)
        action = action[0]
        self.episode_steps+=1
        #print("before command: ",time.time() - self.last_time[0])
        self.pl.torque_command(action,reduce = self.torque_reduce)
       # print("after command: ",time.time() - self.last_time[0])
        #a = choose_action(action_space,Q[0],epsilon = HP.epsilon)
        
        #pl.delta_velocity_command(action_space[a])#update velocity (and steering) and send to simulator. index - index on global path (pl.desired_path)        
       # self.pl.delta_velocity_command(action,self.acc*self.step_time)#*self.step_time
       

    def step(self,action):#get action for gym compatibility

       # action = action[0]
       # self.episode_steps+=1
       # self.pl.torque_command(action,reduce = self.torque_reduce)
       # #a = choose_action(action_space,Q[0],epsilon = HP.epsilon)
        
       # #pl.delta_velocity_command(action_space[a])#update velocity (and steering) and send to simulator. index - index on global path (pl.desired_path)        
       ## self.pl.delta_velocity_command(action,self.acc*self.step_time)#*self.step_time
       # print(action)
        #wait for step time:
        #print("before wait: ",time.time() - self.last_time[0])
       # temp_last_time = self.last_time[0]
        while (not lib.step_now(self.last_time,self.step_time)):# and stop != [True]: #wait for the next step (after step_time)
            time.sleep(0.00001)
        #print("after wait: ",time.time() - temp_last_time)
        #get next state:
        self.pl.simulator.get_vehicle_data()#read data after time step from last action
       # print("after get data: ",time.time() - temp_last_time)
        #local_path = self.pl.get_local_path_vehicle_on_path(send_path = False,num_of_points = self.visualized_points)#num_of_points = visualized_points
        local_path = self.pl.get_local_path(send_path = False,num_of_points = self.visualized_points)#num_of_points = visualized_points
       
        next_state = get_state(self.pl,local_path,self.feature_points,self.distance_between_points,self.max_velocity,self.max_curvature)
        if self.analytic_feature_flag:
            analytic_a = [self.comp_analytic_acceleration(next_state)]
            next_state = analytic_a + next_state
        
        self.dataManager.update_real_path(pl = self.pl,velocity_limit = local_path.analytic_velocity_limit[0],analytic_vel = local_path.analytic_velocity[0]\
            ,curvature = local_path.curvature[0],seed = self.path_seed)
        #self.dataManager.save_additional_data(features = lib.denormalize(next_state,0,30))#pl,features = denormalize(state,0,30),action = a
        if self.end_indication_flag == True:
            end_distance = self.distance_between_points*self.feature_points
        else:
            end_distance = None
        mode = self.pl.check_end(deviation = lib.dist(local_path.position[0][0],local_path.position[0][1],0,0)\
            ,max_roll = self.max_roll,max_pitch = self.max_pitch,end_distance = end_distance)#check if end of the episode 
        #print("roll:",self.pl.simulator.vehicle.angle)
        self.dataManager.roll.append(self.pl.simulator.vehicle.angle[2])
        #get reward:
        reward = get_reward(self.pl.simulator.vehicle.velocity,self.max_velocity,mode)
        #reward = self.pl.simulator.vehicle.velocity / 30 
        if self.episode_steps > self.max_episode_steps:
            mode = 'max_steps'
        #if self.pl.simulator.vehicle.velocity > local_path.analytic_velocity_limit[0]:
        #    mode = 'cross'
        if mode != 'ok':
            #if mode != 'path end' and mode != 'max_steps':
            #    reward = -5
            
            done = True
            #print("done___________________________________")
            if mode != 'kipp' and mode != 'seen_path_end':
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
    def get_analytic_action(self):
        #v1 = self.pl.in_vehicle_reference_path.analytic_velocity[self.pl.main_index]
        #v2 = self.pl.in_vehicle_reference_path.analytic_velocity[self.pl.main_index+1]
        #d = self.pl.in_vehicle_reference_path.distance[self.pl.main_index+1] - self.pl.in_vehicle_reference_path.distance[self.pl.main_index]
        self.pl.simulator.get_vehicle_data()#read data after time step from last action
       # print("after get data: ",time.time() - temp_last_time)
        local_path = self.pl.get_local_path(send_path = False,num_of_points = self.visualized_points)#num_of_points = visualized_points
        v1 = local_path.analytic_velocity[0]
        v2 = local_path.analytic_velocity[1]
        d =local_path.distance[1] - local_path.distance[0]
        
        acc = (v2**2 - v1**2 )/(2*d)#acceleration [m/s^2]
        print("acc:",acc)
        return [np.clip(acc/8,-1,1)]

    def comp_analytic_acceleration(self,pos_state):
        max_distance = self.distance_between_points*self.feature_points
        state_path = classes.Path()
        for j in range(1,len(pos_state)-1,2):
            state_path.position.append([pos_state[j]*max_distance,pos_state[j+1]*max_distance,0.0])
        state_path.comp_distance()
        for i in range(len(state_path.distance)-1):
            if state_path.distance[i+1] - state_path.distance[i]  < 0.01:
                print("dis<0----------------------------------------")
                return -0.7
        #print(state_path.position)
        #with open("state_path", 'w') as f:
        #    for pos in state_path.position:
        #        f.write("%s \t %s\t %s \n" % (pos[0],pos[1],pos[2]))
        result = lib.comp_velocity_limit_and_velocity(state_path,init_vel = pos_state[0]*self.max_velocity, final_vel = 0,reduce_factor = 1.0)
        if result == 1:#computed analytic velocity
            acc = state_path.analytic_acceleration[0]
            return np.clip(acc/8,-1,1)
        else:
            print("vels",pos_state[0]*self.max_velocity,state_path.analytic_velocity_limit[0])
            if pos_state[0]*self.max_velocity > state_path.analytic_velocity_limit[0]:
                print("crossed limit")
            print("cannot compute analytic velocity")
            return -0.7
        

        #error =  3.0 - pos_state[0]*self.max_velocity 
        #return np.clip(error*0.5,-1,1)

        

    def close(self):
        #end all:
        print("close env")
        self.pl.stop_vehicle()
        self.pl.end()
        


        