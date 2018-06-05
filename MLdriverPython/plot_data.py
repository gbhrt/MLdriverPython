import matplotlib.pyplot as plt
import numpy as np
import data_manager1
from hyper_parameters import HyperParameters
import os


def change_failes_value(rewards,mode):
    relative_reward_changed = []
    for i in range(len(mode)):
        if mode[i] == 'kipp' or mode[i] == 'deviate':
            relative_reward_changed.append(None)
        else:
            relative_reward_changed.append(rewards[i])
    return relative_reward_changed


    
HP = HyperParameters()
#dm = data_manager1.DataManager(HP.save_file_path,HP.restore_file_path,HP.restore_flag)
#dm.plot_all()

names = ["final_conv_analytic_1","final_conv_analytic_3"]
episodes_num = 200
dataManager_vec = []
relative_rewards_changed_vec = []
for i in range(len(names)):
    HP.restore_name = names[i]
    HP.save_name = names[i]
    save_path = os.getcwd()+ "\\files\\models\\DDPG\\"+HP.save_name+"\\"
    restore_path = os.getcwd()+ "\\files\\models\\DDPG\\"+HP.restore_name+"\\"
    dataManager_vec.append(data_manager1.DataManager(save_path,restore_path,True))
    relative_rewards_changed_vec.append(np.array(change_failes_value(dataManager_vec[-1].relative_reward,dataManager_vec[-1].episode_end_mode))[:episodes_num])


mean_relative_reward_vec = []
for i in range(episodes_num):
    if all(item[i] == None for item in relative_rewards_changed_vec):
        mean_relative_reward = -2
    else:
        mean_relative_reward = 0
        for relative_rewards_changed in relative_rewards_changed_vec:
            if relative_rewards_changed[i] == None:
                continue
            else:
                mean_relative_reward+=relative_rewards_changed[i]
    
    mean_relative_reward_vec.append(mean_relative_reward/len(dataManager_vec))

plt.scatter(np.array(dataManager_vec[0].run_num)[:episodes_num],relative_rewards_changed_vec[0],alpha = 0.5)
plt.scatter(np.array(dataManager_vec[0].run_num)[:episodes_num],relative_rewards_changed_vec[1],alpha = 0.5)

plt.scatter(np.array(dataManager_vec[0].run_num)[:episodes_num],mean_relative_reward_vec)
plt.show()
