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

def comp_MB_action(net,env,state,acc,steer):
    print("___________________new acc compution____________________________")
    X_dict,abs_pos,abs_ang = predict_lib.initilize_prediction(env,state,acc,steer)
    
    #delta_var = 0.002
    init_var = 0.0#uncertainty of the roll measurment
    const_var = 0.05#roll variance at the future states
    max_plan_roll = env.max_plan_roll
    max_plan_deviation = env.max_plan_deviation
    roll_var = init_var
    pred_vec = [[abs_pos,abs_ang,state['vel_y'],state['roll'],init_var,acc,steer]]
    #stability at the current state:
    roll_flag,dev_flag = predict_lib.check_stability(env,state['path'],0,abs_pos,X_dict['roll'],roll_var = roll_var,max_plan_roll = max_plan_roll,max_plan_deviation = max_plan_deviation)
    if roll_flag == 0 and not dev_flag:
        #predict the next unavoidable state (actions already done):
        X_dict,abs_pos,abs_ang = predict_lib.predict_one_step(net,env,copy.copy(X_dict),abs_pos,abs_ang)
        
        vel = env.denormalize(X_dict['vel_y'],"vel_y")
        index = lib.find_index_on_path(state['path'],abs_pos)    
        steer = predict_lib.steer_policy(abs_pos,abs_ang,state['path'],index,vel)
        X_dict["steer_action"] = env.normalize(steer,"steer_action")
        roll = env.denormalize(X_dict["roll"],"roll")
        #pred_vec.append([abs_pos,abs_ang,vel,roll,roll_var,steer,acc])
        #stability at the next state (unavoidable state):
        roll_flag,dev_flag = predict_lib.check_stability(env,state['path'],index,abs_pos,X_dict['roll'],roll_var = roll_var,max_plan_roll = max_plan_roll,max_plan_deviation = max_plan_deviation)
        
        if roll_flag == 0 and not dev_flag:#first step was Ok
            #max_plan_roll = env.max_plan_roll
            roll_var += const_var
            #integration on the next n steps:
            n = 10
            acc_to_try =[1.0,0.0,-1.0]
            for i,try_acc in enumerate(acc_to_try):
                X_dict["acc_action"] = try_acc
                if i == len(acc_to_try)-1:
                    #max_plan_roll = env.max_plan_roll*1.3
                    roll_var = 0.0
                print("try acc:",try_acc)

                pred_vec_n,roll_flag,dev_flag = predict_lib.predict_n_next1(n,net,env,copy.copy(X_dict),abs_pos,abs_ang,state['path'],max_plan_roll = max_plan_roll,roll_var = roll_var,max_plan_deviation = max_plan_deviation)
                print("roll_flag:",roll_flag,"dev_flag",dev_flag)
                if (roll_flag == 0 and not dev_flag) or i == len(acc_to_try)-1:
                    #check if the emergency policy is save after applying the action on the first step
                    emergency_pred_vec_n,emergency_roll_flag,emergency_dev_flag = predict_lib.predict_n_next1(n,net,env,copy.copy(X_dict),abs_pos,abs_ang,state['path'],emergency_flag = True,max_plan_roll = max_plan_roll,roll_var = roll_var,max_plan_deviation = max_plan_deviation)
                    if emergency_roll_flag == 0 and not emergency_dev_flag:#regular policy and emergency policy are ok:
                        next_acc = try_acc
                        next_steer = steer
                        emergency_action = False
                        break
                #emergency_pred_vec_n,emergency_roll_flag,emergency_dev_flag = [],0,False
                #if (roll_flag == 0 and not dev_flag):
                    
                #    next_acc = try_acc
                #    next_steer = steer
                #    break
                #else the tried action wasn't ok, and must try again
            #pred_vec.append([abs_pos,abs_ang,vel,roll,roll_var,next_steer,next_acc])#the second prediction, the first is the initial state
            #emergency_pred_vec = pred_vec+emergency_pred_vec_n
            #pred_vec+=pred_vec_n
            if (emergency_roll_flag != 0 or emergency_dev_flag) and (roll_flag != 0 or dev_flag):#both regular and emergency failed:
                print("no solution! will fail!!!")

            if (emergency_roll_flag != 0 or emergency_dev_flag) and (roll_flag == 0 and not dev_flag):
                print("solution only for the regular policy - emergency fail")
                next_acc = acc_to_try[-1]#can only be the last
                next_steer = steer
                emergency_action = False

        else:# first step will cause fail
            print("unavoidable fail - after first step")
            #emergency_pred_vec = pred_vec#save just initial state and sirst step
            emergency_pred_vec_n = []
            pred_vec_n = []


        #emergency_roll_flag,emergency_dev_flag = True,True#for the case that already failed

        if (roll_flag != 0 or dev_flag):# and (emergency_roll_flag == 0 and not emergency_dev_flag):#if just emergency is OK:
            next_steer = predict_lib.emergency_steer_policy()# steering from the next state, send it to the vehicle at the next state
            next_acc = predict_lib.emergency_acc_policy()
            emergency_action = True
            print("emergency policy is executed!")

        pred_vec.append([abs_pos,abs_ang,vel,roll,init_var,next_acc,next_steer])#the second prediction, the first is the initial state
        emergency_pred_vec = pred_vec+emergency_pred_vec_n
        pred_vec+=pred_vec_n

    else:#initial state is already unstable
        print("already failed - before first step")
        #emergency_pred_vec = pred_vec#save just initial state
        next_steer = predict_lib.emergency_steer_policy()# steering from the next state, send it to the vehicle at the next state
        next_acc = predict_lib.emergency_acc_policy()
        emergency_action = True
        emergency_pred_vec = pred_vec

    planningData = classes.planningData()
    np_pred_vec = np.array(pred_vec)
    np_emergency_pred_vec = np.array(emergency_pred_vec)

    planningData.vec_planned_roll.append(np_pred_vec[:,3])
    planningData.vec_planned_roll_var.append(np_pred_vec[:,4])
    planningData.vec_emergency_planned_roll.append(np_emergency_pred_vec[:,3])
    planningData.vec_emergency_planned_roll_var.append(np_emergency_pred_vec[:,4])
    planningData.vec_planned_vel.append(np_pred_vec[:,2])
    planningData.vec_emergency_planned_vel.append(np_emergency_pred_vec[:,2])
    planningData.vec_planned_acc.append(np_pred_vec[:,5])
    planningData.vec_planned_steer.append(np_pred_vec[:,6])
    planningData.vec_emergency_planned_acc.append(np_emergency_pred_vec[:,5])
    planningData.vec_emergency_planned_steer.append(np_emergency_pred_vec[:,6])
    planningData.vec_emergency_action.append(emergency_action)


    planningData.vec_path.append(state['path'])
    planningData.vec_predicded_path.append(np_pred_vec[:,0])
    planningData.vec_emergency_predicded_path.append(np_emergency_pred_vec[:,0])

    #return next_acc,next_steer,pred_vec,emergency_pred_vec,roll_flag,dev_flag,emergency_action
    return next_acc,next_steer,planningData,roll_flag,dev_flag

#class ValuesAndNames:
#    values = []
#    names = []


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

        X_n = len(trainHP.vehicle_ind_data) + 2 # + steer-action, desired roll
        Y_n = 1 #acc
        self.AccNet,_ = keras_model.create_model(X_n,Y_n,trainHP.alpha)#model_based_network(X_n,Y_n,trainHP.alpha)

        X_n = len(trainHP.vehicle_ind_data) + 2 # + acc-action, desired roll
        Y_n = 1#steer
        self.SteerNet,_ = keras_model.create_model(X_n,Y_n,trainHP.alpha)

    def restore_all(self,restore_file_path):
        name = "tf_model"
        path = restore_file_path +name+"\\"
        file_name =  path + "TransNet.ckpt"
        self.TransNet.load_weights(file_name)
        file_name =  path + "AccNet.ckpt"
        self.AccNet.load_weights(file_name)
        file_name =  path + "SteerNet.ckpt"
        self.SteerNet.load_weights(file_name)
        print("networks restored")

    def save_all(self,save_file_path):
        #self.TransNet.save_model(save_file_path)
        name = "tf_model"
        path = save_file_path +name+"\\"
        pathlib.Path(path).mkdir(parents=True, exist_ok=True) 

        file_name =  path+"TransNet.ckpt "
        #file_name =  path+name+".ckpt "
        self.TransNet.save_weights(file_name)
        file_name =  path+"AccNet.ckpt"
        self.AccNet.save_weights(file_name)
        file_name =  path + "SteerNet.ckpt"
        self.SteerNet.save_weights(file_name)


class TrainHyperParameters:
    def __init__(self):
        self.num_of_runs = 5000
        self.alpha = 0.0001# #learning rate
        self.batch_size = 64
        self.replay_memory_size = 100000
        self.train_num = 100# how many times to train in every step
        self.run_random_num = 'inf'
        self.vehicle_ind_data = OrderedDict([('vel_y',0),('steer',1), ('roll',2)])  

class Agent:# includes the networks, policies, replay buffer, learning hyper parameters
    def __init__(self,HP):
        self.HP = HP
        self.trainHP = TrainHyperParameters()

        self.Replay = agent_lib.Replay(self.trainHP.replay_memory_size)
        if self.HP.restore_flag:
            self.Replay.restore(self.HP.restore_file_path)

        #define all nets:
        self.nets = Nets(self.trainHP)
        if self.HP.restore_flag:
            self.nets.restore_all(self.HP.restore_file_path)
        self.trainShared = shared.trainShared()
        return
    def save(self):
        self.nets.save_all(self.HP.save_file_path)
        self.Replay.save(self.HP.save_file_path)

    def start_training(self):
        #train the networks
        if self.HP.train_flag:
            self.trainThread = train_thread.trainThread(self.nets,self.Replay,self.trainHP,self.HP,self.trainShared,self.nets.transgraph)
            self.trainThread.start()

    def stop_training(self):
        self.trainShared.train = False
        time.sleep(1.0)
        self.trainShared.request_exit = True
        while not self.trainShared.exit:
            print("waiting for exit train thread")
            time.sleep(0.1)
        print("exit from train thread")


    def comp_action(self,state,acc,steer,env):
        self.trainShared.algorithmIsIn.clear()#indicates that are ready to take the lock
        with self.trainShared.Lock:
            self.trainShared.algorithmIsIn.set()
            return comp_MB_action(self.nets.TransNet,env,state,acc,steer)


    def add_to_replay(self,state,acc,steer,done,time_error,fail):
        self.trainShared.algorithmIsIn.clear()#indicates that are ready to take the lock
        with self.trainShared.Lock:
            self.trainShared.algorithmIsIn.set()
            #replay memory: [vehicle-state, rel_pos,action,done]
            self.Replay.add(copy.deepcopy((state.Vehicle.values,state.Vehicle.rel_pos+[state.Vehicle.rel_ang],[acc,steer],done,time_error,fail)))#      
              
            #X,Y_ =env.create_XY_([state],[[acc,steer]],[next_state])#normalized data
            #dict_X = env.X_to_X_dict(X[0])
            #dict_Y_ = env.Y_to_Y_dict(Y_[0])
            #for name in env.copy_Y_to_X_names:
            #    dict_Y_[name] -= dict_X[name]
            #X = env.dict_X_to_X(dict_X)
            #Y_ = env.dict_Y_to_Y(dict_Y_)
            #self.Replay.add(copy.deepcopy((X,Y_,done,fail)))# 

    def get_state(self,env_state):#take from env what is nedded - the only connection to env.
        S = State()
        S.env = env_state['path']#local path
        S.Vehicle.rel_pos = [env_state['rel_pos_x'],env_state['rel_pos_y']]
        S.Vehicle.rel_ang = env_state['rel_ang']
        S.Vehicle.values = []
        for feature in self.trainHP.vehicle_ind_data.keys():
            S.Vehicle.values.append(env_state[feature])
        return S
