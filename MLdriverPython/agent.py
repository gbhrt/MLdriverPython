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
from DDPG_net import DDPG_network
import sys

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
    def __init__(self,trainHP): 
        X_n = len(trainHP.vehicle_ind_data)+2# + acc-action, steer-action
        Y_n = len(trainHP.vehicle_ind_data) + 3 #+dx, dy, dang
        #self.TransNet = model_based_network(X_n,Y_n,trainHP.alpha)
        self.TransNet,self.transgraph = keras_model.create_model(X_n,Y_n,trainHP.alpha)
        self.TransNet._make_predict_function()
        #X_n = len(trainHP.vehicle_ind_data) + 2 # + steer-action, desired roll
        #Y_n = 1 #acc
        #self.AccNet,_ = keras_model.create_model(X_n,Y_n,trainHP.alpha)#model_based_network(X_n,Y_n,trainHP.alpha)

        X_n = len(trainHP.vehicle_ind_data) + 2 # + acc-action, desired roll
        Y_n = 1#steer
        self.SteerNet,_ = keras_model.create_model(X_n,Y_n,trainHP.alpha)
        self.restore_error = False
    def restore_all(self,restore_file_path,name):
        try:
            path = restore_file_path +name+"/"
            file_name =  path + "TransNet.ckpt"
            self.TransNet.load_weights(file_name)
            #file_name =  path + "AccNet.ckpt"
            #self.AccNet.load_weights(file_name)
            file_name =  path + "SteerNet.ckpt"
            self.SteerNet.load_weights(file_name)
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

        file_name =  path+"TransNet.ckpt "
        #file_name =  path+name+".ckpt "
        self.TransNet.save_weights(file_name)
        #file_name =  path+"AccNet.ckpt"
        #self.AccNet.save_weights(file_name)
        file_name =  path + "SteerNet.ckpt"
        self.SteerNet.save_weights(file_name)

class MF_Net:#define the input and outputs to networks, and the nets itself.
    def __init__(self,trainHP,HP,envData): 
        net = DDPG_network(trainHP.state_n,trainHP.action_n,\
                trainHP.MF_alpha_actor,trainHP.MF_alpha_critic,tau = trainHP.tau,seed = HP.seed[0],feature_data_n = envData.feature_data_num, conv_flag = trainHP.conv_flag) 
        if HP.restore_flag:
            net.restore(HP.restore_file_path)#cannot restore - return true

class TrainHyperParameters:
    def __init__(self):
        self.MF_policy_flag = False
        self.num_of_runs = 5000
        self.alpha = 0.0001# #learning rate
        self.batch_size = 64
        self.replay_memory_size = 100000
        self.train_num = 100# how many times to train in every step
        self.run_random_num = 'inf'
        self.vehicle_ind_data = OrderedDict([('vel_y',0),('steer',1), ('roll',2)])  
        self.plan_roll = 0.03
        #self.emergency_plan_roll = 0.07
        self.target_tolerance = 0.02
        self.min_dis = 0.5#or precentage
        self.max_plan_deviation = 10
        self.max_plan_roll = 0.1
        self.init_var = 0.0#uncertainty of the roll measurment
        self.one_step_var = 0.01
        self.const_var = 0.05#roll variance at the future states, constant because closed loop control?
       
        #self.emergency_const_var = 0.05

        self.emergency_action_flag = False
        self.emergency_steering_type = 1#1 - stright, 2 - 0.5 from original steering, 3-steer net
        

        self.max_cost = 100
        self.rollout_n = 10
        if self.MF_policy_flag:
            self.MF_alpha_actor = 0.0001
            self.MF_alpha_critic = 0.001
            self.tau = 0.001 
            self.conv_flag = False
            self.state_n = len(self.vehicle_ind_data)+3
            self.action_n = 2



 

class Agent:# includes the networks, policies, replay buffer, learning hyper parameters
    def __init__(self,HP,envData = None):
        self.HP = HP
        self.trainHP = TrainHyperParameters()
        self.trainHP.emergency_action_flag = HP.emergency_action_flag
        self.trainHP.emergency_steering_type = HP.emergency_steering_type#1 - stright, 2 - 0.5 from original steering, 3-steer net

        self.Replay = agent_lib.Replay(self.trainHP.replay_memory_size)
        if self.HP.restore_flag:
            self.Replay.restore(self.HP.restore_file_path)

        #define all nets:
        if self.trainHP.MF_policy_flag:
            self.MF_net = MF_Net(self.trainHP,HP,envData)
        self.nets = Nets(self.trainHP)
        if self.HP.restore_flag:
            self.nets.restore_all(self.HP.restore_file_path,self.HP.net_name)
        self.trainShared = shared.trainShared()
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
        planningData.vec_planned_roll.append([StateVehicle.values[2] for StateVehicle in StateVehicle_vec])
        planningData.vec_planned_vel.append([StateVehicle.values[0] for StateVehicle in StateVehicle_vec])
        planningData.vec_planned_acc.append([action[0] for action in actions_vec])
        planningData.vec_planned_steer.append([action[1] for action in actions_vec])
        if StateVehicle_emergency_vec is not None:
            planningData.vec_emergency_predicded_path.append([StateVehicle.abs_pos for StateVehicle in StateVehicle_emergency_vec])
            planningData.vec_emergency_planned_roll.append([StateVehicle.values[2] for StateVehicle in StateVehicle_emergency_vec])
            planningData.vec_emergency_planned_vel.append([StateVehicle.values[0] for StateVehicle in StateVehicle_emergency_vec])
            planningData.vec_emergency_planned_acc.append([action[0] for action in actions_emergency_vec])
            planningData.vec_emergency_planned_steer.append([action[1] for action in actions_emergency_vec])
            planningData.vec_emergency_action.append(emergency_action)
        return planningData


    def comp_action(self,state,acc,steer):#env
        self.trainShared.algorithmIsIn.clear()#indicates that are ready to take the lock
        with self.trainShared.Lock:
            self.trainShared.algorithmIsIn.set()
            acc,steer,StateVehicle_vec,actions_vec,StateVehicle_emergency_vec,actions_emergency_vec,emergency_action = act.comp_MB_action(self.nets,state,acc,steer,self.trainHP)
            planningData = self.convert_to_planningData(state.env,StateVehicle_vec,actions_vec,StateVehicle_emergency_vec,actions_emergency_vec,emergency_action)
            return acc,steer,planningData #act.comp_MB_action(self.nets.TransNet,env,state,acc,steer)
    
    def get_MF_action(self,state):
        self.targetPoint = target_point.comp_targetPoint(self.nets,state,self.trainHP)#tmp - must be computed independly from steps
        MF_state = convert_to_MF_state(state,self.targetPoint)
        action  = net_stabilize.get_actions([MF_state])[0]
        return action[0],action[1]

    def add_to_replay(self,state,acc,steer,done,time_error,fail):
        self.trainShared.algorithmIsIn.clear()#indicates that are ready to take the lock
        with self.trainShared.Lock:
            self.trainShared.algorithmIsIn.set()
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
