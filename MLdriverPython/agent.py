import MB_action_lib
import agent_lib
import train_thread
import shared
import copy
import time
from model_based_net import model_based_network
import keras_model
from collections import OrderedDict
import pathlib
import classes
import predict_lib
import library as lib
import numpy as np
import target_point
import actions_for_given_path as act
#from DDPG_net import DDPG_network
import sys
import direct_method
import json

def convert_to_MF_state(state,targetPoint):
    return state.Vehicle.values + state.Vehicle.rel_pos+[state.Vehicle.rel_ang]+[targetPoint.abs_pos]+[targetPoint.vel]


class VehicleState:
    values = []
    abs_pos = []#x,y
    abs_ang = 0#angle
    rel_pos = []#dx,dy,
    rel_ang = 0 # dangle


class State:
    def __init__(self):
        self.Vehicle = VehicleState()#velocity, roll, current steering, ... 
        self.env = []# path, obstacles...
        return

class Nets:#define the input and outputs to networks, and the nets itself.
    def __init__(self,trainHP,trans_net_active = True,steer_net_active = True,acc_net_active = False): 
        self.trans_net_active,self.steer_net_active,self.acc_net_active = trans_net_active,steer_net_active,acc_net_active
        if self.trans_net_active:
            X_n = len(trainHP.vehicle_ind_data)+2# + acc-action, steer-action
            Y_n = len(trainHP.vehicle_ind_data) + 3 #+dx, dy, dang
            #self.TransNet = model_based_network(X_n,Y_n,trainHP.alpha)
            self.TransNet,self.transgraph = keras_model.create_model(X_n,Y_n,trainHP.alpha,seperate_nets = True, normalize = trainHP.normalize_flag,mean= trainHP.features_mean,var = trainHP.features_var)
        
            self.TransNet._make_predict_function()
        if self.acc_net_active:
            X_n = len(trainHP.vehicle_ind_data) + 2 # + steer-action, desired roll
            Y_n = 1 #acc
            self.AccNet,_ = keras_model.create_model(X_n,Y_n,trainHP.alpha)#model_based_network(X_n,Y_n,trainHP.alpha)
        if self.steer_net_active:
            X_n = len(trainHP.vehicle_ind_data) + 2 # + acc-action, desired roll
            Y_n = 1#steer
            self.SteerNet,_ = keras_model.create_model(X_n,Y_n,trainHP.alpha,seperate_nets = False)
        self.restore_error = False
    def restore_all(self,restore_file_path,name):
        try:
            path = restore_file_path +name+"/"
            if self.trans_net_active:
                self.TransNet.load_weights(path + "TransNet.ckpt")
            if self.acc_net_active:
                self.AccNet.load_weights(path + "AccNet.ckpt")
            if self.steer_net_active:
                self.SteerNet.load_weights(path + "SteerNet.ckpt")
            print("networks restored")
            self.restore_error = False
        except:
            print('cannot restore net',sys.exc_info()[0])
            #raise
            self.restore_error = True

    def save_all(self,save_file_path,name):
        #self.TransNet.save_model(save_file_path)
        
        path = save_file_path +name+"/"
        pathlib.Path(path).mkdir(parents=True, exist_ok=True) 
        if self.trans_net_active:
            self.TransNet.save_weights(path+"TransNet.ckpt ")
        if self.acc_net_active:
            self.AccNet.save_weights(path+"AccNet.ckpt")
        if self.steer_net_active:
            self.SteerNet.save_weights(path + "SteerNet.ckpt")

class MF_Net:#define the input and outputs to networks, and the nets itself.
    def __init__(self,trainHP,HP,envData): 
        net = DDPG_network(trainHP.state_n,trainHP.action_n,\
                trainHP.MF_alpha_actor,trainHP.MF_alpha_critic,tau = trainHP.tau,seed = HP.seed[0],feature_data_n = envData.feature_data_num, conv_flag = trainHP.conv_flag) 
        if HP.restore_flag:
            net.restore(HP.restore_file_path)#cannot restore - return true

class TrainHyperParameters:
    def __init__(self,HP):
        self.MF_policy_flag = False
        self.direct_predict_active = False
        self.direct_constrain = True #stabilization constrain computed by direct model (centrpetal force limit) or roll constrain
        self.update_var_flag = True 
        self.num_of_runs = 5000
        self.alpha = 0.0001# #learning rate
        self.batch_size = 64
        self.replay_memory_size = 100000
        self.train_num = 100# how many times to train in every step
        self.run_random_num = 'inf'
        self.vehicle_ind_data = OrderedDict([('vel_y',0),('steer',1)])  #, ('angular_vel_z',4)  , ('roll',2) ('vel_x',3),  ('angular_vel_z',4)
        self.normalize_flag = False
        if self.normalize_flag == True:
            self.features_mean = [7,0,0,0,0]#input feature + action
            self.features_var = [7,0.5,0.05,0.7,0.7]
        else:
            self.features_mean = None
            self.features_var = None
        self.direct_stabilize = HP.direct_stabilize
        self.plan_roll = 0.03
        #self.emergency_plan_roll = 0.07
        self.target_tolerance = 0.02
        self.min_dis = 0.5#or precentage
        self.max_plan_deviation = 10
        self.max_plan_roll = 0.1
        self.init_var = 0.0#uncertainty of the roll measurment
        if self.update_var_flag:
            self.one_step_var =1.0
            self.const_var =1.0
        else:
            #0.5 not move. 0.2 < 0.5 of VOD. 0.1 =0.85 of VOD. 0 1+-0.05 of VOD com height = 1.7
            self.one_step_var =10#0.04# 0.02 is good
            self.const_var = 1000.0#0.05#roll variance at the future states, constant because closed loop control?
        self.prior_safe_velocity = 0.02#if the velocity is lower than this value - it is priori Known that it is OK to accelerate
        self.stabilize_factor = 1.0
        #self.emergency_const_var = 0.05
      
        self.emergency_action_flag = HP.emergency_action_flag
        self.emergency_steering_type = HP.emergency_steering_type#1 - stright, 2 - 0.5 from original steering, 3-steer net
     
        

        self.max_cost = 100
        self.rollout_n = 13#10
        if self.MF_policy_flag:
            self.MF_alpha_actor = 0.0001
            self.MF_alpha_critic = 0.001
            self.tau = 0.001 
            self.conv_flag = False
            self.state_n = len(self.vehicle_ind_data)+3
            self.action_n = 2
            


class PlanningState:
    def __init__(self,trainHP):
        self.trust_T = 20
        self.var = [trainHP.init_var]+[min(trainHP.init_var+trainHP.one_step_var*n,trainHP.const_var ) for n in range(1,trainHP.rollout_n+10)]
        self.last_emergency_action_active = False


 

class Agent:# includes the networks, policies, replay buffer, learning hyper parameters
    def __init__(self,HP,envData = None,trans_net_active = True,steer_net_active = True,acc_net_active = False,one_step_var = None):
        self.HP = HP
        self.trainHP = TrainHyperParameters(self.HP)
        self.var_name = "var"
        if one_step_var is not None:
            #self.trainHP.one_step_var = one_step_var
            self.trainHP.const_var = one_step_var
            
        self.planningState = PlanningState(self.trainHP)
        self.Replay = agent_lib.Replay(self.trainHP.replay_memory_size)
        if self.HP.restore_flag:
            self.Replay.restore(self.HP.restore_file_path)

        self.Direct = direct_method.directModel(self.trainHP)
        #define all nets:
        if self.trainHP.MF_policy_flag:
            self.MF_net = MF_Net(self.trainHP,HP,envData)
        self.nets = Nets(self.trainHP,trans_net_active,steer_net_active,acc_net_active)
        #self.train_nets = Nets(self.trainHP,trans_net_active,steer_net_active,acc_net_active)
        self.trainShared = shared.trainShared()
        if self.HP.restore_flag:
            #self.train_nets.restore_all(self.HP.restore_file_path,self.HP.net_name)
            #self.copy_nets()
            self.nets.restore_all(self.HP.restore_file_path,self.HP.net_name)
        
        return
    def save_nets(self):
        with self.trainShared.Lock:
            self.nets.save_all(self.HP.save_file_path,self.HP.net_name)
    def save(self):
        with self.trainShared.Lock:
            self.nets.save_all(self.HP.save_file_path,self.HP.net_name)
        self.Replay.save(self.HP.save_file_path)

    def start_training(self):
        print("start_training", self.HP.train_flag)
        #train the networks
        if self.HP.train_flag:
            #self.trainThread = train_thread.trainThread(self.train_nets,self.Replay,self.trainHP,self.HP,self.trainShared)
            self.trainThread = train_thread.trainThread(self.nets,self.Replay,self.trainHP,self.HP,self.trainShared)
            self.trainThread.start()
            #if self.trainHP.MF_policy_flag:
            #    self.MFtrainThread = train_thread.trainThread(self.nets,self.Replay,self.trainHP,self.HP,self.trainShared,self.nets.transgraph)
            #    self.MFtrainThread.start()

    def stop_training(self):
        self.trainShared.train = False
        time.sleep(1.0)
        self.trainShared.request_exit = True
        #while not self.trainShared.exit:
        #    print("waiting for exit train thread")
        #    time.sleep(0.1)
        print("exit from train thread")


    def convert_to_planningData(self,state_env,StateVehicle_vec,actions_vec,StateVehicle_emergency_vec = None,actions_emergency_vec = None,emergency_action = False):#,targetPoint_vec = []
        planningData = classes.planningData()
      
        planningData.vec_path.append(state_env[0])
        #print("state.env:",state.env.position)
        planningData.vec_predicded_path.append([StateVehicle.abs_pos for StateVehicle in StateVehicle_vec])
        #planningData.vec_planned_roll.append([StateVehicle.values[self.trainHP.vehicle_ind_data["roll"]] for StateVehicle in StateVehicle_vec])
        planningData.vec_planned_roll.append([[0] for StateVehicle in StateVehicle_vec])
        planningData.vec_planned_vel.append([StateVehicle.values[self.trainHP.vehicle_ind_data["vel_y"]] for StateVehicle in StateVehicle_vec])
        planningData.vec_planned_acc.append([action[0] for action in actions_vec])
        planningData.vec_planned_steer.append([action[1] for action in actions_vec])
        if StateVehicle_emergency_vec is not None:
            planningData.vec_emergency_predicded_path.append([StateVehicle.abs_pos for StateVehicle in StateVehicle_emergency_vec])
            #planningData.vec_emergency_planned_roll.append([StateVehicle.values[self.trainHP.vehicle_ind_data["roll"]] for StateVehicle in StateVehicle_emergency_vec])
            planningData.vec_emergency_planned_roll.append([[0] for StateVehicle in StateVehicle_emergency_vec])
            planningData.vec_emergency_planned_vel.append([StateVehicle.values[self.trainHP.vehicle_ind_data["vel_y"]] for StateVehicle in StateVehicle_emergency_vec])
            planningData.vec_emergency_planned_acc.append([action[0] for action in actions_emergency_vec])
            planningData.vec_emergency_planned_steer.append([action[1] for action in actions_emergency_vec])
            planningData.vec_emergency_action.append(emergency_action)
        return planningData


    def comp_action(self,state,acc,steer):#env
        self.trainShared.algorithmIsIn.clear()#indicates that are ready to take the lock
        with self.trainShared.Lock:
            self.trainShared.algorithmIsIn.set()
            with self.nets.transgraph.as_default():  
                acc,steer,StateVehicle_vec,actions_vec,StateVehicle_emergency_vec,actions_emergency_vec,emergency_action = act.comp_MB_action(self.nets,state,acc,steer,self.trainHP,
                                                                                                                                              planningState = self.planningState,
                                                                                                                                              Direct = self.Direct if (self.trainHP.direct_stabilize or self.trainHP.direct_predict_active or self.trainHP.direct_constrain) else None
                                                                                                                                              )
        self.planningState.last_emergency_action_active = emergency_action
        planningData = self.convert_to_planningData(state.env,StateVehicle_vec,actions_vec,StateVehicle_emergency_vec,actions_emergency_vec,emergency_action)
        return acc,steer,planningData #act.comp_MB_action(self.nets.TransNet,env,state,acc,steer)
    
    def get_MF_action(self,state):
        self.targetPoint = target_point.comp_targetPoint(self.nets,state,self.trainHP)#tmp - must be computed independly from steps
        MF_state = convert_to_MF_state(state,self.targetPoint)
        action  = net_stabilize.get_actions([MF_state])[0]
        return action[0],action[1]

    def add_to_replay(self,state,acc,steer,done,time_error,fail):
        #self.trainShared.algorithmIsIn.clear()#indicates that are ready to take the lock
        with self.trainShared.ReplayLock:
            #self.trainShared.algorithmIsIn.set()
            #replay memory: [vehicle-state, rel_pos,action,done]
            self.Replay.add(copy.deepcopy((state.Vehicle.values,state.Vehicle.rel_pos+[state.Vehicle.rel_ang],[acc,steer],done,time_error,fail)))#      
              


    def get_state(self,env_state):#take from env what is nedded - the only connection to env.
        S = State()
        path = env_state['path']
        path.position = np.array(path.position)
        S.env = [path,0]#[env_state['path'],0]#local path
        S.Vehicle.rel_pos = [env_state['rel_pos_x'],env_state['rel_pos_y']]
        #print("rel_pos:",S.Vehicle.rel_pos)
        S.Vehicle.rel_ang = env_state['rel_ang']
        S.Vehicle.abs_pos = [0.0,0.0]
        S.Vehicle.abs_ang = 0.0
        S.Vehicle.values = []
        for feature in self.trainHP.vehicle_ind_data.keys():
            S.Vehicle.values.append(env_state[feature])

        return S
    def copy_nets(self):
        t = time.clock()
        with self.trainShared.Lock:
            if self.nets.trans_net_active:
                self.nets.TransNet.set_weights(self.train_nets.TransNet.get_weights()) 
            if self.nets.steer_net_active:
                self.nets.SteerNet.set_weights(self.train_nets.SteerNet.get_weights()) 
            if self.nets.acc_net_active:
                self.nets.AccNet.set_weights(self.train_nets.AccNet.get_weights()) 
        print("copy time:", time.clock() - t)



    def update_episode_var(self,episode_lenght):
        episode_lenght = 2000
        with self.trainShared.ReplayLock:
            episode_lenght = min(episode_lenght,len(self.Replay.memory))
            episode_replay_memory = self.Replay.memory[-episode_lenght:]

        with self.trainShared.Lock:
            n = self.trainHP.rollout_n
            n_state_vec,n_state_vec_pred,n_pos_vec,n_pos_vec_pred,n_ang_vec,n_ang_vec_pred = predict_lib.get_all_n_step_states(self.Direct if self.trainHP.direct_predict_active else self.nets.TransNet,
                                                                                                                                self.trainHP,
                                                                                                                                episode_replay_memory, n)
        #compute variance/abs_error for all raw features
        # var_vec,mean_vec,pos_var_vec,pos_mean_vec,ang_var_vec,ang_mean_vec = predict_lib.comp_var(self, n_state_vec,n_state_vec_pred,n_pos_vec,n_pos_vec_pred,n_ang_vec,n_ang_vec_pred,type = "mean_error")
        # val_var = var_vec[1]
        # #roll_var = [0]+[var[self.trainHP.vehicle_ind_data["roll"]] for var in var_vec]+[var_vec[-1][self.trainHP.vehicle_ind_data["roll"]]]*(20-len(var_vec)-1)#0.1
        # roll_var = [0]+[var[self.trainHP.vehicle_ind_data["roll"]] for var in var_vec]+[0.1]*(20-len(var_vec)-1)#0.1
        # print("roll_var:",roll_var)
        # self.planningState.var = roll_var

        #compute the variance/abs_error of the centripetal acceleration. 
        var_vec = predict_lib.comp_ac_var(self, n_state_vec,n_state_vec_pred,type = "var")#"mean_error"
        var_vec = [0]+var_vec+[1.0]*(20-len(var_vec)-1)#0.1
        print("var_vec:",var_vec)
        self.planningState.var = var_vec

    def save_var(self):
        path = self.HP.save_file_path +self.var_name+"/"
        pathlib.Path(path).mkdir(parents=True, exist_ok=True) 
        print("var path: ", path+"var.txt")
        try: 
            with open(path+"var.txt", 'w') as f:
                #json.dump((self.run_num,self.train_num,self.rewards,self.lenght,self.relative_reward, self.episode_end_mode,self.path_seed,self.paths ),f)
                json.dump((self.planningState.var),f)

            print("var saved")            
        except:
            print("cannot save var", sys.exc_info()[0])
    def load_var(self):
        path = self.HP.restore_file_path +self.var_name+"/"
        print("var path: ", path+"var.txt")
        try:
            with open(path+"var.txt", 'r') as f:
                #self.run_num,self.train_num,self.rewards,self.lenght,self.relative_reward, self.episode_end_mode,self.path_seed,self.paths = json.load(f)#,self.paths
                self.planningState.var = json.load(f)#,self.paths

            print("var restored")
            return False
        except:
            print ("cannot restore var:", sys.exc_info()[0])
            return True



        
        

