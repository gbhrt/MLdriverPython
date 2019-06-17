import threading

import environment1
import data_manager1
import hyper_parameters 
from DDPG_net import DDPG_network
import saftey_DDPG_algorithm
import tkinker_gui
import shared


def run(guiShared,HP,dataManager):
    envData = environment1.OptimalVelocityPlannerData(env_mode = "SDDPG")
    dataManager = data_manager1.DataManager(HP.save_file_path,HP.restore_file_path,HP.restore_flag)

    net = DDPG_network(envData.observation_space.shape[0],envData.action_space.shape[0],envData.action_space.high[0],\
        HP.alpha_actor,HP.alpha_critic,HP.alpha_analytic_actor,HP.alpha_analytic_critic,tau = HP.tau,seed = HP.seed[0],feature_data_n = envData.feature_data_num, conv_flag = HP.conv_flag)  
    if HP.restore_flag:
        net.restore(HP.restore_file_path)#cannot restore - return true

    net_stabilize = DDPG_network(envData.observation_space.shape[0],envData.action_space.shape[0],envData.action_space.high[0],\
        HP.alpha_actor,HP.alpha_critic,HP.alpha_analytic_actor,HP.alpha_analytic_critic,tau = HP.tau,seed = HP.seed[0],feature_data_n = envData.feature_data_num, conv_flag = HP.conv_flag)  
    #if HP.restore_flag:
    #    net_stabilize.restore(HP.restore_file_path)#cannot restore - return true
    #train agent on simulator
    env = environment1.OptimalVelocityPlanner(dataManager,env_mode="SDDPG")
    if env.opened:     
        saftey_DDPG_algorithm.train(env,HP,net,net_stabilize,dataManager,guiShared = guiShared)


class programThread (threading.Thread):
   def __init__(self,guiShared,HP,dataManager):
        threading.Thread.__init__(self)
        self.guiShared = guiShared
        self.HP = HP
        self.dataManager = dataManager
      
   def run(self):
        print ("Starting " + self.name)
        run(self.guiShared,self.HP,self.dataManager)
        print ("Exiting " + self.name)


if __name__ == "__main__": 
    guiShared = shared.guiShared()
    HP = hyper_parameters.SafteyHyperParameters()
    dataManager = data_manager1.DataManager(HP.save_file_path,HP.restore_file_path,HP.restore_flag)

    # Create new thread
    programThread = programThread(guiShared,HP,dataManager)
    programThread.start()

    if HP.gui_flag:
        #start the gui:
        tkinker_gui.TkGui(guiShared)
    while(not guiShared.exit):
        time.sleep(1)
        continue

    print ("Exiting Main Thread")

