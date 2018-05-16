import enviroment1
import data_manager1
from hyper_parameters import HyperParameters
from DDPG_net import DDPG_network
import numpy as np

HP = HyperParameters()
dataManager = data_manager1.DataManager(HP.save_file_path,HP.restore_file_path,HP.restore_flag)
env = enviroment1.OptimalVelocityPlanner(dataManager)
net = DDPG_network(env.observation_space.shape[0],env.action_space.shape[0],env.action_space.high[0],HP.alpha_actor,HP.alpha_critic,tau = HP.tau,seed = HP.seed)  
net.restore(HP.restore_file_path)
state = env.reset()
state = [0.1 for _ in state]
    
print("state:",state)
for vel in range (15):
    state[0] = float(vel)
    a = [[0]]
    Qa = net.get_Qa(np.reshape(state, (1, env.observation_space.shape[0])),a)[0][0]
    Q0 = net.get_Qa(np.reshape(state, (1, env.observation_space.shape[0])),[[0]])[0][0]
    Q1 = net.get_Qa(np.reshape(state, (1, env.observation_space.shape[0])),[[1.0]])[0][0]
    Qneg1 = net.get_Qa(np.reshape(state, (1, env.observation_space.shape[0])),[[-1.0]])[0][0]
    print("vel:",vel,"Qa:",Qa,"Q0:",Q0,"Q1",Q1,"Qneg1",Qneg1)

env.close()