from planner import Planner
import numpy as np
import library as lib
import classes
import copy
import random
import os
import time
#import coll
from environment_lib import *
class ActionSpace:
    shape = []
    high = []
class ObservationSpace:
    shape = []
    range = []


class OptimalVelocityPlannerData:
    def __init__(self,env_mode = 'model_based',X_names = None,Y_names = None):#env_mode = 'DDPG' ,env_mode = 'model_based' 'model_based'

        self.env_mode = env_mode#'model_based' #'model_based'  'DDPG'#env_mode 
        self.stop_flag = False
        self.mode = 'ok'

        self.analytic_feature_flag = False
        self.roll_feature_flag = False
        self.vehicle_data_features = True
        self.wheels_vel_feature_flag = False
        self.end_indication_flag = False
        self.lower_bound_flag = False

        self.update_max_roll_flag = False
        self.roll_vec=[]#save all roll angles during one episode
        self.max_episode_steps = 100#100#200#
        self.feature_points = 25#60 # number of points on the path in the state +(1*self.end_indication_flag)
        self.feature_data_num = 1+(8*self.vehicle_data_features)  + (1*self.analytic_feature_flag)+ (1*self.roll_feature_flag) + (5*self.wheels_vel_feature_flag)# number of data in state (vehicle velocity, analytic data...)
        self.distance_between_points = 1.0 #meter 1.0
        self.path_length = 9000#self.feature_points*self.distance_between_points
        self.step_time = 0.2
        if env_mode == 'SDDPG' or env_mode == 'DDPG_target':
            self.action_space_n = 2
        elif env_mode == 'SDDPG_pure_persuit':
            self.action_space_n = 1
        else:
            self.action_space_n = 1

        self.visualized_points = int(self.feature_points/0.05) + 10 #how many points show on the map and length of local path

        self.max_deviation =  100#10 # [m] if more then maximum - end episode 
        self.max_plan_deviation = 10
        
        self.max_velocity_x = 30
        self.max_velocity_y = 30
        self.max_velocity_z = 30
        self.max_angular_velocity_x = 6
        self.max_angular_velocity_y = 6
        self.max_angular_velocity_z = 6
        self.max_acc_x = 1.38# 0-100 kmh in 20 sec. 1.5 # [m/s^2]  need to be more than maximum acceleration in real
        self.max_acc_y = 1.38# 0-100 kmh in 20 sec. 1.5 # [m/s^2]  need to be more than maximum acceleration in real
        self.max_acc_z = 1.38# 0-100 kmh in 20 sec. 1.5 # [m/s^2]  need to be more than maximum acceleration in real
        self.max_angular_acc_x = 100
        self.max_angular_acc_y = 100
        self.max_angular_acc_z = 100

        self.max_pitch = 0.3#0.3
        self.max_roll = 0.7#0.05#0.3# last 0.2 #0.1 is good
        self.max_slip = 10
        self.max_plan_slip = 0.1
        self.max_plan_roll = 0.05
        self.max_steering = 0.7
        self.max_d_steering = 0.1
        self.max_wheel_vel = 60# rad/sec. in unity limited to 5720 deg/sec 
        self.torque_reduce = 1.0 # 0.2
        self.reduce_factor = 1.0
        self.max_episode_length = 500
        self.action_space = ActionSpace()
        self.action_space.shape = [self.action_space_n]
        self.action_space.high = [1.0]#self.acc*self.step_time] #, torque in Nm
        self.observation_space = ObservationSpace()
        
        if self.env_mode == 'DDPG' or self.env_mode == 'SDDPG' or self.env_mode == 'SDDPG_pure_persuit':
            self.X_n = self.feature_data_num +  2*self.feature_points #vehicle state points on the path (distance)
            self.observation_space.range = [[0,self.max_velocity_y],
                                [-self.max_steering,self.max_steering],
                                [-self.max_roll,self.max_roll],
                                [-self.max_steering,self.max_steering],
                                [-1.0,1.0]]#vel,steer,roll,steer_comand,acc_comand
            self.observation_space.shape = [self.X_n]
        elif self.env_mode == 'DDPG_target':
            self.X_n = self.feature_data_num+3 #vel,x taget,y target,vel target
            self.observation_space.shape = [self.X_n]
        elif self.env_mode == 'model_based':
            self.max_min_values = {'vel_x':[0,self.max_velocity_x],
                                 #'vel_y':[0,self.max_velocity_y],
                                 'vel_y':[-self.max_acc_y,self.max_acc_y],
                                 'vel_z':[0,self.max_velocity_z],
                                 'vel':[0,self.max_velocity_y],#max, min, number
                                 'angular_vel_x':[-self.max_angular_velocity_x,self.max_angular_velocity_x],
                                 'angular_vel_y':[-self.max_angular_velocity_y,self.max_angular_velocity_y],
                                 'angular_vel_z':[-self.max_angular_velocity_z,self.max_angular_velocity_z],
                                 'acc_x':[-self.max_acc_x,self.max_acc_x],
                                 'acc_y':[-self.max_acc_y,self.max_acc_y],
                                 'acc_z':[-self.max_acc_z,self.max_acc_z],
                                 'angular_acc_x':[-self.max_angular_acc_x,self.max_angular_acc_x],
                                 'angular_acc_y':[-self.max_angular_acc_y,self.max_angular_acc_y],
                                 'angular_acc_z':[-self.max_angular_acc_z,self.max_angular_acc_z],
                                 #'steer':[-self.max_steering,self.max_steering],
                                 'steer':[-self.max_d_steering,self.max_d_steering],
                                 'roll':[-self.max_roll,self.max_roll],
                                 'steer_action':[-self.max_steering,self.max_steering],
                                 'acc_action':[-1.0,1.0],
                                 'rel_pos_x':[-5,5],
                                 'rel_pos_y':[0,5],
                                 'rel_pos_z':[-5,5],
                                 'rel_ang':[-1.0,1.0]
                                   }
            self.features_numbers = {'vel_x':1,
                                     'vel_y':1,
                                    'vel_z':1,
                                    'angular_vel_x':1,
                                    'angular_vel_y':1,
                                    'angular_vel_z':1,
                                    'acc_x':1,
                                    'acc_y':1,
                                    'acc_z':1,
                                    'angular_acc_x':1,
                                    'angular_acc_y':1,
                                    'angular_acc_z':1,
                                    "steer":1,
                                    "roll":1,
                                    'wheel_n_vel':2,
                                    'rel_pos_x':1,
                                    'rel_pos_y':1,
                                    'rel_pos_z':1,
                                    "rel_ang":1,
                                    "acc_action":1,
                                    "steer_action":1}

            #self.X_names = ["vel_x","vel_y","vel_z","angular_vel","acc","angular_acc","steer","roll","steer_action","acc_action"]
            #self.Y_names = ["vel_x","vel_y","vel_z","angular_vel","acc","angular_acc","steer","roll","rel_pos","rel_ang"]#,'wheel_n_vel'
            #self.copy_Y_to_X_names = ["vel_x","vel_y","vel_z","angular_vel","acc","angular_acc","steer","roll"]
            if X_names is None:
                self.X_names = ["vel_y","steer","roll","acc_action","steer_action"]
                self.Y_names = ["vel_y","steer","roll","rel_pos_x","rel_pos_y","rel_ang"]
                self.copy_Y_to_X_names = ["vel_y","steer","roll"]

                #self.X_names = ["vel_x","vel_y","vel_z",
                #                "angular_vel_x","angular_vel_y","angular_vel_z",
                #                "acc_x","acc_y","acc_z",
                #                "angular_acc_x","angular_acc_y","angular_acc_z",
                #                "steer","roll","steer_action","acc_action"]
                #self.Y_names = ["vel_x","vel_y","vel_z",
                #                "angular_vel_x","angular_vel_y","angular_vel_z",
                #                "acc_x","acc_y","acc_z",
                #                "angular_acc_x","angular_acc_y","angular_acc_z",
                #                "steer","roll",
                #                "rel_pos_x","rel_pos_y",
                #                "rel_ang"]
                #self.copy_Y_to_X_names = ["vel_x","vel_y","vel_z",
                #                           "angular_vel_x","angular_vel_y","angular_vel_z",
                #                           "acc_x","acc_y","acc_z",
                #                           "angular_acc_x","angular_acc_y","angular_acc_z",
                #                         "steer","roll"]
            else:                     
                self.X_names = X_names
                self.Y_names = Y_names

        


            self.observation_space.range = []
            for name in self.X_names:
                for _ in range(self.features_numbers[name]):
                    self.observation_space.range.append(self.max_min_values[name])

            
  
            self.X_n = 0 
            for name in self.X_names:
                self.X_n+=self.features_numbers[name]

            #self.Y_n = self.X_n + 3 #add rel x rel y rel ang, remove steering command and acceleration command  
            self.Y_n = 0
            for name in self.Y_names:
                self.Y_n+=self.features_numbers[name]
            self.observation_space.shape = [self.X_n]
        
        self.max_curvature = 0.12

        self.lt = time.clock()

    def X_to_X_dict(self,X):
        X_dict = {}
        j = 0
        for name in self.X_names:
            if self.features_numbers[name] == 1:
                X_dict[name] = X[j]
                j+=1
            else:
                x = []
                for i in range(self.features_numbers[name]):
                    x.append(X[j])
                    j+=1
                X_dict[name] = x
        return X_dict
    def Y_to_Y_dict(self,Y):
        Y_dict = {}
        j = 0
        for name in self.Y_names:
            if self.features_numbers[name] == 1:
                Y_dict[name] = Y[j]
                j+=1
            else:
                y = []
                for i in range(self.features_numbers[name]):
                    y.append(Y[j])
                    j+=1
                Y_dict[name] = y
        return Y_dict
    def dict_X_to_X(self,X_dict):
        X = []
        for name in self.X_names:
            Xi = []
            num = self.features_numbers[name]
            if num == 1:
                X.append(X_dict[name])
            else:
                for i in range(num):
                    X.append(X_dict[name][i])
        return X
    def dict_Y_to_Y(self,Y_dict):
        Y = []
        for name in self.Y_names:
            Yi = []
            Y.append(Y_dict[name])
        return Y

    def normalize(self,name_val, name):
        if self.features_numbers[name] == 1:
            norm = lib.normalize_value(name_val,self.max_min_values[name][0],self.max_min_values[name][1])
        else:
            norm = []
            for i in range(self.features_numbers[name]):
                norm.append(lib.normalize_value(name_val[i],self.max_min_values[name][0],self.max_min_values[name][1]))

        return norm
    def denormalize_dict(self,X):
        denorm_X = {}
        for key,val in X.items():
            denorm_X[key] = self.denormalize(val,key)
        return denorm_X
    def denormalize(self,name_val, name):
        if self.features_numbers[name] == 1:
            denorm = lib.denormalize_value(name_val,self.max_min_values[name][0],self.max_min_values[name][1])
        else:
            denorm = []
            for i in range(self.features_numbers[name]):
                denorm.append(lib.denormalize_value(name_val[i],self.max_min_values[name][0],self.max_min_values[name][1]))
        return denorm

    def create_X(self,state,a):
        X = []
        for i in range(len(state)):
            Xi = []
            for name in self.X_names[:-2]:#X names includes actions hence [:-3]
                if not isinstance(state[i][name], list):
                    Xi += [self.normalize(state[i][name],name)]#
                else:
                    Xi += self.normalize(state[i][name],name)# 

            Xi += [self.normalize(np.clip(a[i][0],-1.0,1.0),"acc_action") , self.normalize(a[i][1],"steer_action")]#steer,acc

            X.append(Xi)
        return X
    def create_X_1(self,state,a):
        X = []
        for i in range(len(state)):
            Xi = []
            for name in self.X_names[:-2]:#X names includes actions hence [:-3]
                if not isinstance(state[i][name], list):
                    Xi += [state[i][name]]#     
                else:
                    Xi += state[i][name]# 

            Xi += [a[i][1] ,np.clip(a[i][0],-1.0,1.0)]#steer,acc

            X.append(Xi)
        return X
    def create_XY_(self,state,a,next_state):
        X = self.create_X(state,a)
        Y_ = []
        for i in range(len(state)):
            Yi = []
            for name in self.Y_names:
                if not isinstance(next_state[i][name], list):
                    Yi += [self.normalize(next_state[i][name],name)]#
                else:
                    Yi += self.normalize(next_state[i][name],name)#
            Y_.append(Yi)

        return X,Y_

    def create_XY_1(self,state,a,next_state):
        X = self.create_X_1(state,a)
        Y_ = []
        for i in range(len(state)):
            Yi = []
            for name in self.Y_names:
                if not isinstance(next_state[i][name], list):
                    Yi += [next_state[i][name]]#
                else:
                    Yi += next_state[i][name]#
            Y_.append(Yi)

        return X,Y_

        

class OptimalVelocityPlanner(OptimalVelocityPlannerData):
    def __init__(self,dataManager,env_mode="model_based"):
        
        super().__init__(env_mode = env_mode)
        self.dataManager = dataManager
       
        ####
        self.path_name = 'paths//circle_r7_json.txt'#"paths//straight_path_limit_json.txt"     #long random path: path3.txt  #long straight path: straight_path.txt
        self.path_source = "create_random" #"create"#  "regular" #"create_random" #saved_random"

        self.reset_count = 0
        self.reset_every = 5
   
        self.pl = Planner("torque")
        self.opened = self.pl.simulator.connected

        self.path_seed = None
        self.last_pos = self.pl.simulator.vehicle.position#first time init abs position and angle - updated in the function
        self.last_ang = self.pl.simulator.vehicle.angle
        return

    def get_state(self):
        if self.env_mode == 'DDPG' or self.env_mode == 'SDDPG' or self.env_mode == 'SDDPG_pure_persuit':
            
            state = get_ddpg_state(self.pl,self.local_path,self.feature_points,self.distance_between_points,self.max_velocity_y,self.max_curvature)
            if self.roll_feature_flag:
                state = [self.pl.simulator.vehicle.angle[2]/self.max_roll]+state
            if self.wheels_vel_feature_flag:
                state = [self.pl.simulator.vehicle.steering/self.max_steering]+state
                state = [wheel.angular_vel/self.max_wheel_vel for wheel in self.pl.simulator.vehicle.wheels]+state
            if self.analytic_feature_flag:
                analytic_a = [self.comp_analytic_acceleration(state)]
                state = analytic_a + state
            if self.vehicle_data_features:
                state = [self.pl.simulator.vehicle.angle[2]/self.max_roll]+\
                [self.pl.simulator.vehicle.angle[0]/self.max_roll]+\
                [self.pl.simulator.vehicle.steering] +\
                self.pl.simulator.vehicle.angular_velocity+\
                [self.pl.simulator.vehicle.velocity[0]]+\
                [self.pl.simulator.vehicle.velocity[2]]+\
                state



        if self.env_mode == 'DDPG_target':
            state = get_ddpg_target_state(self.pl,self.local_path,self.feature_points,self.distance_between_points,self.max_velocity_y,self.max_curvature)
            if self.roll_feature_flag:
                state = [self.pl.simulator.vehicle.angle[2]/self.max_roll]+state
            if self.wheels_vel_feature_flag:
                state = [self.pl.simulator.vehicle.steering/self.max_steering]+state
                state = [wheel.angular_vel/self.max_wheel_vel for wheel in self.pl.simulator.vehicle.wheels]+state
            if self.vehicle_data_features:
                state = [self.pl.simulator.vehicle.angle[2]/self.max_roll]+\
                [self.pl.simulator.vehicle.angle[0]/self.max_roll]+\
                [self.pl.simulator.vehicle.steering] +\
                self.pl.simulator.vehicle.angular_velocity+\
                [self.pl.simulator.vehicle.velocity[0]]+\
                [self.pl.simulator.vehicle.velocity[2]]+\
                state
        if self.env_mode == 'model_based':
            #self.last_pos = self.pl.simulator.vehicle.position#first time init abs position and angle - updated in the function
            #self.last_ang = self.pl.simulator.vehicle.angle
            state = get_model_based_state(self.pl,self.last_pos,self.last_ang,self.local_path)
        return state

    def reset(self,seed = None):
        self.stop_flag = False
  #     self.lt = 0 #tmp
        if (self.reset_count % self.reset_every == 0 and self.reset_count > 0): 
            self.error = self.pl.simulator.reset_position()
            if self.error:
                return "error"
         #   print("self.error",self.error)
            self.pl.stop_vehicle()
          #  print("after stop")
        self.episode_steps = 0
        
        self.pl.restart()#not sure
       # print("after restart")
        self.error =  self.pl.simulator.get_vehicle_data()#read data after time step from last action
        if self.error:
            return "error"
       # print("after restart self.error",self.error)
        #if self.path_source == "saved_random" or self.path_source == "create_random":
        if seed == None:
            seed = int.from_bytes(os.urandom(8), byteorder="big")
            self.dataManager.path_seed.append(seed)
   #     else:
            #if len(self.dataManager.path_seed) > path_num:
            #    seed = self.dataManager.path_seed[path_num]
            #else:
            #    print("error - path id not exist")
        if self.reset_count>0 and len(self.roll_vec) > 25 and self.update_max_roll_flag:#not at the first time
            max_episode_roll = max(self.roll_vec[:-25])# dont consider the last roll angles (during 5 sec) because maybe the are not asymtotic stable
            if self.max_plan_roll < max_episode_roll: self.max_plan_roll = max_episode_roll
             

        path_error = self.pl.load_path(self.path_length, self.path_name,self.path_source,seed = seed)#,seed = 1236
        self.path_seed = seed
        if path_error == -1:#if cannot load path
            return "error"
        
        self.pl.new_episode()#compute path in current vehicle position
        #self.dataManager.update_planned_path(self.pl.in_vehicle_reference_path)
        #first state:
        self.local_path = self.pl.get_local_path()#num_of_points = self.visualized_points
        self.error = self.pl.send_desired_path()
        if self.error:
            return "error"
        
        self.last_pos[0] = self.pl.simulator.vehicle.position[0]
        self.last_pos[1] = self.pl.simulator.vehicle.position[1]
        self.last_pos[2] = self.pl.simulator.vehicle.position[2]
        self.last_ang[1] = self.pl.simulator.vehicle.angle[1]
        state = self.get_state()

        #self.dataManager.update_real_path(pl = self.pl,velocity_limit = self.local_path.analytic_velocity_limit[0],analytic_vel = self.local_path.analytic_velocity[0]\
        #    ,curvature = self.local_path.curvature[0],seed = self.path_seed)#update at the first time
        #if self.end_indication_flag:
        #    dis_from_end = self.features_num*self.feature_points - self.pl.in_vehicle_reference_path.distance[self.pl.main_index]
            #self.dataManager.add(('curvature',(self.pl.desired_path.curvature)))
        print("reset done")
        self.reset_count+=1
        self.last_time = [time.clock()]
        return state

    def command(self,action,steer = None):# the algo assume that the action is in the last state, must be close as possible to the end of the
                                #last step command, hence the training is between command and state update (while waiting for step time)
        #action = action[0]
        self.episode_steps+=1
        #print("before command: ",time.time() - self.last_time[0])
        steer_command = self.pl.torque_command(action,steer = steer,reduce = self.torque_reduce)

       # print("after command: ",time.time() - self.last_time[0])
        #a = choose_action(action_space,Q[0],epsilon = HP.epsilon)
        
        #pl.delta_velocity_command(action_space[a])#update velocity (and steering) and send to simulator. index - index on global path (pl.desired_path)        
       # self.pl.delta_velocity_command(action,self.acc*self.step_time)#*self.step_time
        return steer_command

    def step(self,action = None,steer = None,stabilize_flag = False):#get action for gym compatibility
       # self.pl.torque_command(action,reduce = self.torque_reduce)
        #wait for step time:

        #print("step begin time:",time.clock() - self.lt)
        time_step_error = lib.wait_until_end_step(self.last_time,self.step_time)
        self.lt = time.clock()
        #get next state:
        self.error = self.pl.simulator.get_vehicle_data()#read data after time step from last action
        #print("error,get_vehicle_data:",self.error)
        #print("get data time: ",time.clock() - self.lt)
        if self.env_mode == "model_based" and steer is not None:
            self.command(action,steer)#send action immidetlly after get state data. this action is based upon the estimation of the current state
        # that is estimated from the previus state
        #print("command time: ",time.clock() - self.lt)

        self.local_path = self.pl.get_local_path(num_of_points = self.visualized_points)#num_of_points = visualized_points

        next_state = self.get_state()

        self.dataManager.update_real_path(pl = self.pl,velocity_limit = self.local_path.analytic_velocity_limit[0],analytic_vel = self.local_path.analytic_velocity[0]\
            ,curvature = self.local_path.curvature[0],seed = self.path_seed)
        #self.dataManager.save_additional_data(features = lib.denormalize(next_state,0,30))#pl,features = denormalize(state,0,30),action = a
        if self.end_indication_flag == True:
            end_distance = self.distance_between_points*self.feature_points
        else:
            end_distance = None
        #print("before check end: ",time.time() - self.last_time[0])
        deviation = math.copysign( lib.dist(self.local_path.position[0][0],self.local_path.position[0][1],0,0),self.local_path.position[0][0])
        self.mode = self.pl.check_end(deviation = abs(deviation), max_deviation = self.max_deviation,max_roll = self.max_roll,max_pitch = self.max_pitch,end_distance = end_distance)#check if end of the episode 
        #print("roll:",self.pl.simulator.vehicle.angle)
        self.dataManager.roll.append(self.pl.simulator.vehicle.angle[2])
        #self.dataManager.wheels_vel.append(self.pl.simulator.vehicle.wheels_angular_vel)
        self.dataManager.time_stamps.append(self.pl.simulator.vehicle.last_time_stamp) 
        self.dataManager.input_time.append(self.pl.simulator.vehicle.input_time) 
        self.dataManager.step_times.append(self.lt)

        #get reward:
        if self.env_mode == 'SDDPG' or self.env_mode == 'SDDPG_pure_persuit':
            if stabilize_flag:
                reward_stabilize = get_SDDPG_reward_stabilize(self.pl.simulator.vehicle.velocity[1],self.max_velocity_y,self.pl.simulator.vehicle.angle[2],self.mode,deviation)
                #print("reward_stabilize:",reward_stabilize)
                #self.pl.index
            reward = get_SDDPG_reward(self.local_path.angle[0],self.pl.simulator.vehicle.velocity[1],self.max_velocity_y,self.pl.simulator.vehicle.angle[2],self.mode,deviation,steer = self.pl.simulator.vehicle.steering)
        else:
            reward = get_reward(self.pl.simulator.vehicle.velocity[1],self.max_velocity_y,self.mode,steer = self.pl.simulator.vehicle.steering)#,analytic_velocity = self.local_path.analytic_velocity[0])#

        if self.episode_steps > self.max_episode_steps:
            self.mode = 'max_steps'
        #    self.stop_flag = True
        
        #if self.stop_flag and self.pl.simulator.vehicle.velocity[1] < 0.1:
        #    self.mode = 'max_steps'
        if self.mode != 'ok':        
            done = True
        else:
            done = False
        #print("reward", reward, "velocity: ", self.pl.simulator.vehicle.velocity, "mode:", mode)
        info = [self.mode,time_step_error or self.error != 0]
        #print("end step time: ",time.clock() - self.lt)
        if stabilize_flag:
            return next_state, reward,reward_stabilize, done, info
        else:
            return next_state, reward, done, info

    def stop_vehicle_complete(self):
        if (self.mode != 'kipp' and self.mode != 'seen_path_end') or self.error != 0:
            self.pl.stop_vehicle()
            #self.stop_flag = True
            self.error = 0
        if  self.mode == 'kipp': #(i % HP.reset_every == 0 and i > 0) or
            #self.pl.stop_vehicle()
            self.pl.simulator.reset_position()
            self.pl.stop_vehicle()

    def seed(self,seed_int):
        return#TODO set seed in unity
    def render(self):
        return#TODO set unity with / without graphics
    def get_analytic_action(self):
        #v1 = self.pl.in_vehicle_reference_path.analytic_velocity[self.pl.main_index]
        #v2 = self.pl.in_vehicle_reference_path.analytic_velocity[self.pl.main_index+1]
        #d = self.pl.in_vehicle_reference_path.distance[self.pl.main_index+1] - self.pl.in_vehicle_reference_path.distance[self.pl.main_index]
        #self.pl.simulator.get_vehicle_data()#read data after time step from last action
       # print("after get data: ",time.time() - temp_last_time)
        self.local_path = self.pl.get_local_path(num_of_points = self.visualized_points)#num_of_points = visualized_points
        v1 = self.pl.simulator.vehicle.velocity[1]#self.local_path.analytic_velocity[0]
        v2 = self.local_path.analytic_velocity[1]#next velocity
        d =self.local_path.distance[1] - self.local_path.distance[0]
        
        acc = (v2**2 - v1**2 )/(2*d)#acceleration [m/s^2]
       # print("acc:",acc)
        return [np.clip(acc/8,-1,1)]
        #return [np.clip(acc*100,-1,1)]

    def comp_analytic_acceleration(self,pos_state):
        print("vel:",pos_state[0]*self.max_velocity_y)
        max_distance = self.distance_between_points*self.feature_points
        state_path = classes.Path()
        for j in range(1,len(pos_state)-1,2):
            state_path.position.append([pos_state[j]*max_distance,pos_state[j+1]*max_distance,0.0])
        state_path.comp_distance()
        for i in range(len(state_path.distance)-1):
            if state_path.distance[i+1] - state_path.distance[i]  < 0.01:
                #print("dis<0----------------------------------------")
                return -0.7
        #print(state_path.position)
        #with open("state_path", 'w') as f:
        #    for pos in state_path.position:
        #        f.write("%s \t %s\t %s \n" % (pos[0],pos[1],pos[2]))
        result = lib.comp_velocity_limit_and_velocity(state_path,init_vel = pos_state[0]*self.max_velocity_y, final_vel = 0,reduce_factor = self.reduce_factor)
        max_acc = lib.cf.max_acc
        #print("max_acc:",max_acc)
        if result == 1:#computed analytic velocity
            v1 = state_path.analytic_velocity[0]#must be the same as current velocity
            #for i, t in enumerate (state_path.analytic_time):
            #    if t>self.step_time:
            #        break
            for i in range(len (state_path.analytic_time)):
                if state_path.analytic_time[i] > self.step_time:
                    break

            
            v2 = state_path.analytic_velocity[i]
            d = state_path.distance[i]# - path.distance[0]
            #print("i:",i, "time:",state_path.analytic_time[i],"v2:",v2,"d:",d)
            if d < 0.01:
                print("_____________________________________________________________________________")
            acc = (v2**2 - v1**2 )/(2*d)#acceleration [m/s^2]
            acc_tmp = state_path.analytic_acceleration[0]

            #print("acc",acc/max_acc,"acc_tmp:",acc_tmp/max_acc)
            return np.clip(acc/max_acc,-1,1)
            #return np.clip(acc/8,-1,1)
        else:
            #print("vels",pos_state[0]*self.max_velocity_y,state_path.analytic_velocity_limit[0])
            if pos_state[0]*self.max_velocity_y > state_path.analytic_velocity_limit[0]:
                #print("crossed limit")
                v1 = pos_state[0]*self.max_velocity_y
                d = v1*self.step_time
                #print("v1:",v1,"d:",d)
                for i in range(len (state_path.distance)):
                    if state_path.distance[i] > d:
                        break

                v2 = state_path.analytic_velocity_limit[i]
                #print("i:",i,"v2:",v2)
                acc = (v2**2 - v1**2 )/(2*d)
                return np.clip(acc/max_acc,-1,1)
            #print("cannot compute analytic velocity")
            return -1.0
        


    def comp_const_vel_acc(self,des_vel):
        kp=1.0
        error =  (des_vel - self.pl.simulator.vehicle.velocity[1])*kp
        return np.clip(error*0.5,-1,1)


    def comp_analytic_acc_compare(self):
        if self.pl.main_index+10 < len(self.pl.in_vehicle_reference_path.analytic_velocity):
            vel = self.pl.in_vehicle_reference_path.analytic_velocity[self.pl.main_index+10]
        else:
            vel = self.pl.in_vehicle_reference_path.analytic_velocity[-1]
        kp = 100
            #constant velocity:
        error =  (vel - self.pl.simulator.vehicle.velocity[1])*kp
        acc = np.clip(error*0.5,-1,1)
        return acc

    def comp_analytic_velocity(self,pos_state):
        max_distance = self.distance_between_points*self.feature_points
        state_path = classes.Path()
        for j in range(1,len(pos_state)-1,2):
            state_path.position.append([pos_state[j]*max_distance,pos_state[j+1]*max_distance,0.0])
        state_path.comp_distance()
        for i in range(len(state_path.distance)-1):
            if state_path.distance[i+1] - state_path.distance[i]  < 0.01:
                #print("dis<0----------------------------------------")
                return -1.0
        #print(state_path.position)
        #with open("state_path", 'w') as f:
        #    for pos in state_path.position:
        #        f.write("%s \t %s\t %s \n" % (pos[0],pos[1],pos[2]))
        result = lib.comp_velocity_limit_and_velocity(state_path,init_vel = pos_state[0]*self.max_velocity_y, final_vel = 0,reduce_factor = 0.8)
        if result == 1:#computed analytic velocity
            vel = state_path.analytic_velocity[0]
            return vel
        else:
            print("vels",pos_state[0]*self.max_velocity_y,state_path.analytic_velocity_limit[0])
            if pos_state[0]*self.max_velocity_y > state_path.analytic_velocity_limit[0]:
                print("crossed limit")
            print("cannot compute analytic velocity")
            return 0.0

    def check_lower_bound(self,state):
        vel = self.pl.simulator.vehicle.velocity[1]
        analytic_vel = state.analytic_velocity[0]

    def comp_steer(self):
        local_path = self.pl.get_local_path(num_of_points = self.visualized_points)
        steer_target = lib.comp_steer_target(local_path,self.pl.simulator.vehicle.velocity[1])
        steer = lib.comp_steer_local(steer_target)
        return steer
    def close(self):
        #end all:
        print("close env")
        self.pl.stop_vehicle()
        self.pl.end()
        


        