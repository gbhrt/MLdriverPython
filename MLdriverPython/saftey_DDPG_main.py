import threading
from time import sleep
import random
import os

import environment1
import data_manager1
import hyper_parameters 
from DDPG_net import DDPG_network
import saftey_DDPG_algorithm
import model_based_algorithm
import tkinker_gui
import shared
import agent
import copy


def run_all(HP,guiShared):
    envData = environment1.OptimalVelocityPlannerData(env_mode = HP.env_mode)
    if HP.env_mode == "model_based" and HP.gui_flag:
        guiShared.max_roll = envData.max_plan_roll
        guiShared.max_time = envData.step_time*envData.max_episode_steps+5#add time for braking
    names_vec = []
    #names_vec.append([['MB_R_1','MB_R_2','MB_R_3','MB_R_4','MB_R_5'],'MB_R',None])
    #var_constants_vec = [0.01*i for i in range(1,10)]
    #names = ['VOD_var_check_'+str(var_constant) for var_constant in var_constants_vec]
    #names_vec.append([names,'VOD',None])
    names_vec.append([['MB_R_var'],'MB_R',None])
    #names_vec.append([['also_steer1'],'REVO',50000])#trained MF acc and steer policy
    random.seed(0)
    HP.seed = random.sample(range(1000),101)#the 101 path is not executed
    HP.evaluation_every = 999999999
    for names,method,training_num in names_vec:
        HP.analytic_action = False
        if method =='REVO':
            envData.analytic_feature_flag = False
            HP.add_feature_to_action  = False
        if method =='REVO+A':
            envData.analytic_feature_flag = False
            HP.add_feature_to_action  = True
        if method =='REVO+F':
            envData.analytic_feature_flag = True
            HP.add_feature_to_action  = False

        #if method =='VOD':
        #    envData.analytic_feature_flag = False
        #    HP.add_feature_to_action  = False
        #    HP.analytic_action = True
        #    HP.evaluation_flag = True
        #    #reduce_vec = [0.02*i for i in range(1,10)]
        #    reduce_vec = [0.175]
        #    HP.train_flag = False
        #    HP.always_no_noise_flag = True
        #    HP.restore_flag = False
        #    HP.num_of_runs = 100
        #    HP.save_every_train_number = 500000000

        if method == 'VOD':          
            for name,var_constant in zip(names,var_constants_vec):
                HP.restore_name = name
                HP.restore_file_path = HP.folder_path+HP.restore_name+"/"
                HP.save_name = name
                HP.save_file_path = HP.folder_path+HP.save_name+"/"
                HP.num_of_runs = 5#100
                #HP.reduce_vel = reduce
                run_data = []
                Agent = agent.Agent(HP,trans_net_active = True, steer_net_active = False,one_step_var = var_constant)
                Agent.trainHP.direct_predict_active = True
                Agent.trainHP.update_var_flag = False
                dataManager = data_manager1.DataManager(HP.save_file_path,HP.restore_file_path,restore_flag =False,save_name = 'data_manager_0')

                #train agent on simulator
                env = environment1.OptimalVelocityPlanner(dataManager,env_mode=HP.env_mode)
                if env.opened:     
                    Agent.trainHP.num_of_runs = HP.num_of_runs
                    model_based_algorithm.train(env,HP,Agent,dataManager,guiShared,global_train_count = 0)
                continue

        if method == "MB_R":#model based regular - without stabilization
            HP.max_steps = 300#1000
            HP.emergency_action_flag = False
            HP.emergency_steering_type = 1#1 - stright, 2 - 0.5 from original steering, 3-steer net
        elif method == "MB_S":
            HP.max_steps = 1000
            HP.emergency_action_flag = True
            HP.emergency_steering_type = 1#1 - stright, 2 - 0.5 from original steering, 3-steer net
        description = method 
        run_data = ["envData.analytic_feature_flag: "+str(envData.analytic_feature_flag), 
            "HP.add_feature_to_action: "+str(HP.add_feature_to_action),
            "reduce_vel: "+str(HP.reduce_vel),
            "seed: "+str(HP.seed),
            description]
        HP.save_every_train_number = 100#5000
        
        for evalutaion_flag in [True]:#False,
            HP.evaluation_flag = evalutaion_flag

            for name in names:
                HP.restore_name = name
                HP.restore_file_path = HP.folder_path+HP.restore_name+"/"
                HP.save_name = name#"save_movie"#
                HP.save_file_path = HP.folder_path+HP.save_name+"/"
                
                if HP.evaluation_flag:
                    HP.train_flag = False
                    HP.always_no_noise_flag = True
                    HP.restore_flag = True
                    HP.num_of_runs = 100

                    if training_num is  None:
                        nums = [HP.save_every_train_number*j for j in range(0,50)]
                    else:
                        nums = [training_num]
                    for i in nums:
                        print("num:",i)
                        #dataManager = data_manager1.DataManager(HP.save_file_path,HP.restore_file_path,restore_flag = True,restore_name = 'data_manager_'+str(i))
                        #if not dataManager.error and len(dataManager.episode_end_mode) >= HP.num_of_runs-10: 
                        #    print(name,'data_manager_'+str(i)+' exist')
                        #    continue
                        print("HP.num_of_runs:",HP.num_of_runs)
                        print("evaluation on episode:",i)
                        dataManager = data_manager1.DataManager(HP.save_file_path,HP.restore_file_path,restore_flag =False,save_name = 'data_manager_'+str(i))
                        if HP.env_mode == "model_based":
                            restore_error = run_train_MB(HP,dataManager,envData,index = i)
                        else:
                            restore_error = run_train(HP,dataManager,envData,index = i)

                        if restore_error:
                            print(name,"cannot restore:",'tf_model_'+str(i))
                            continue
                else:#not evaluation
                    HP.train_flag = True
                    HP.always_no_noise_flag = False
                    HP.restore_flag = False
                    HP.num_of_runs = 100#limited by training number 
                    

                    dataManager = data_manager1.DataManager(HP.save_file_path,HP.restore_file_path,HP.restore_flag)
                    dataManager.run_data = run_data
                    dataManager.save_run_data()
                    if HP.env_mode == "model_based":
                        run_train_MB(HP,dataManager,envData)
                    else:
                        run_train(HP,dataManager,envData)
                    #
            sleep(3)

def run_train_MB(HP,dataManager,envData,index = None):

    global_train_count = 0
    if HP.restore_flag:
        if HP.evaluation_flag:
            HP.net_name = 'tf_model_'+str(index)
            Agent = agent.Agent(HP,trans_net_active = True, steer_net_active = False)
            Agent.trainHP.update_var_flag = False
            #tmp fix for getting saved var from datamanager
            Agent.var_name = 'var_'+str(index)
            Agent.load_var()
            if Agent.nets.restore_error:#cannot restore - return true
                return True
            

        else:#try to restore the last one
            nums = [HP.save_every_train_number*j for j in range(1,11)]
            found_flag = False
            for i in nums:
                
                HP.net_name = 'tf_model_'+str(i)
                Agent = agent.Agent(HP,trans_net_active = True, steer_net_active = False)

                if Agent.nets.restore_error:#False if OK
                    found_flag = True

                elif found_flag:
                    global_train_count = i-HP.save_every_train_number
                    break
            print("train,global_train_count:",global_train_count)
    else:
        Agent = agent.Agent(HP,trans_net_active = True, steer_net_active = False)

    #train agent on simulator
    env = environment1.OptimalVelocityPlanner(dataManager,env_mode=HP.env_mode)
    if env.opened:     
        Agent.trainHP.num_of_runs = HP.num_of_runs
        model_based_algorithm.train(env,HP,Agent,dataManager,guiShared,global_train_count = global_train_count)
    return False

def run_train(HP,dataManager,envData,index = None):
    global_train_count = 0
    net = DDPG_network(envData.observation_space.shape[0],envData.action_space.shape[0],\
        HP.alpha_actor,HP.alpha_critic,HP.alpha_analytic_actor,HP.alpha_analytic_critic,tau = HP.tau,seed = HP.seed[0],feature_data_n = envData.feature_data_num, conv_flag = HP.conv_flag)  

    if HP.stabilize_flag:
        net_stabilize = DDPG_network(envData.observation_space.shape[0],envData.action_space.shape[0],\
            HP.alpha_actor,HP.alpha_critic,HP.alpha_analytic_actor,HP.alpha_analytic_critic,tau = HP.tau,seed = HP.seed[0],feature_data_n = envData.feature_data_num, conv_flag = HP.conv_flag)  
    
    if HP.restore_flag:
        if HP.evaluation_flag:
            if net.restore(HP.restore_file_path,name = 'tf_model_'+str(index)):#cannot restore - return true
                return True
            if HP.stabilize_flag:
                if net_stabilize.restore(HP.restore_file_path,name = 'tf_model_stabilize_'+str(index)):#cannot restore - return true
                    return True

        else:#try to restore the last one
            nums = [HP.save_every_train_number*j for j in range(1,21)]
            found_flag = False
            for i in nums:
                restored = not net.restore(HP.restore_file_path,name = 'tf_model_'+str(i))
                if HP.stabilize_flag:
                    restored = restored and not net_stabilize.restore(HP.restore_file_path,name = 'tf_model_stabilize_'+str(i))
                if restored:#False if OK
                    found_flag = True

                elif found_flag:
                    global_train_count = i-HP.save_every_train_number
                    break
            print("train,global_train_count:",global_train_count)

    #train agent on simulator
    env = environment1.OptimalVelocityPlanner(dataManager,env_mode=HP.env_mode)
    if env.opened:     
        if HP.stabilize_flag:
            saftey_DDPG_algorithm.train(env,HP,net,dataManager,net_stabilize = net_stabilize,guiShared = guiShared,global_train_count = global_train_count)
        else:
            saftey_DDPG_algorithm.train(env,HP,net,dataManager,guiShared = guiShared,global_train_count = global_train_count)    
    return False



def run(HP,guiShared = None):
    #Agent = agent.Agent(HP)

    envData = environment1.OptimalVelocityPlannerData(env_mode = HP.env_mode)
    dataManager = data_manager1.DataManager(HP.save_file_path,HP.restore_file_path,HP.restore_flag)

    net = DDPG_network(envData.observation_space.shape[0],envData.action_space.shape[0],\
        HP.alpha_actor,HP.alpha_critic,HP.alpha_analytic_actor,HP.alpha_analytic_critic,tau = HP.tau,seed = HP.seed[0],feature_data_n = envData.feature_data_num, conv_flag = HP.conv_flag)  
    if HP.restore_flag:
        net.restore(HP.restore_file_path)#cannot restore - return true
    if HP.stabilize_flag:
        net_stabilize = DDPG_network(envData.observation_space.shape[0],envData.action_space.shape[0],\
            HP.alpha_actor,HP.alpha_critic,HP.alpha_analytic_actor,HP.alpha_analytic_critic,tau = HP.tau,seed = HP.seed[0],feature_data_n = envData.feature_data_num, conv_flag = HP.conv_flag)  
        if HP.restore_flag:
            net_stabilize.restore(HP.restore_file_path,name = 'tf_model_stabilize')#cannot restore - return true
    
    #train agent on simulator
    env = environment1.OptimalVelocityPlanner(dataManager,env_mode=HP.env_mode)
    if env.opened:     
        if HP.stabilize_flag:
            saftey_DDPG_algorithm.train(env,HP,net,dataManager,net_stabilize = net_stabilize,guiShared = guiShared)
        else:
            saftey_DDPG_algorithm.train(env,HP,net,dataManager,guiShared = guiShared)


class programThread (threading.Thread):
    def __init__(self,guiShared,HP):
        threading.Thread.__init__(self)
        self.guiShared = guiShared
        self.HP = HP
        
      
    def run(self):
        print ("Starting " + self.name)
        #run(self.HP,self.guiShared)
        run_all(self.HP,self.guiShared)
        print ("Exiting " + self.name)

        
        


if __name__ == "__main__": 
    algo_type = "MB"#"MB"#"SDDPG"

    if algo_type == "SDDPG":
        HP = hyper_parameters.SafteyHyperParameters()
    elif algo_type == "DDPG":
        HP = hyper_parameters.HyperParameters()
    elif algo_type == "MB":
        HP = hyper_parameters.ModelBasedHyperParameters()
    else:
        print("error - algo_type not exist")

    guiShared = shared.guiShared(HP.env_mode) if HP.gui_flag else None
    

    # Create new thread
    programThread = programThread(guiShared,HP)
    programThread.start()

    if HP.gui_flag:
        #start the gui:
        tkinker_gui.TkGui(guiShared)
        while(not guiShared.exit):
            sleep(1)
            continue

    print ("Exiting Main Thread")




