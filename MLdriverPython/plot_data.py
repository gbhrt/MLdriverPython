import matplotlib.pyplot as plt
import numpy as np
import data_manager1
from hyper_parameters import HyperParameters
import os
import library as lib
import copy
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


"""
from second computer:
final_conv_analytic_2 - old rewards
final_conv_analytic_4 - old rewards
final_conv_2 - old rewards
final_conv_new_reward_2 - new relative reward >1000
final_conv_new_reward_4 - new relative reward >200

this computer:
final_conv_1 - converted relative reward
final_conv_3 - new relative reward - path saved
final_conv_analytic_1 - converted relative reward maybe no analytic
final_conv_analytic_3 - converted relative reward
final_conv_analytic_2 - converted relative reward
final_conv_analytic_4 - converted relative reward
final_conv_new_reward_1 - new relative reward >1000
final_conv_new_reward_3 - new relative reward >200
final_conv_analytic_new_reward_2 - new relative reward >200
final_conv_analytic_new_reward_4 - new relative reward >200
"""

def plot_rewards(names,shape):
    episodes_num = 50
    dataManager_vec = []
    relative_rewards_changed_vec = []
    for i in range(len(names)):
        HP.restore_name = names[i]
        HP.save_name = names[i]
        save_path = os.getcwd()+ "\\files\\models\\DDPG\\"+HP.save_name+"\\"
        restore_path = os.getcwd()+ "\\files\\models\\DDPG\\"+HP.restore_name+"\\"
        dataManager_vec.append(data_manager1.DataManager(save_path,restore_path,True))
        relative_rewards_changed_vec.append(np.array(change_failes_value(dataManager_vec[-1].relative_reward,dataManager_vec[-1].episode_end_mode)))#[:episodes_num]

        
    #mean_relative_reward_vec = []
    #mean_relative_reward_ind = []
    #for i in range(episodes_num):
    #    if all(item[i] == None for item in relative_rewards_changed_vec):
    #        mean_relative_reward = 0.0
            
    #    else:
    #        mean_relative_reward = 0.0
    #        mean_train_num = 0.0
    #        count = 0
    #        for j in range(len(relative_rewards_changed_vec)):
    #            if relative_rewards_changed_vec[j][i] == None:
    #                continue
    #            else:
    #                mean_relative_reward+=relative_rewards_changed_vec[j][i]
    #                mean_train_num+= dataManager_vec[j].train_num[i]
    #                count+=1
    #        mean_relative_reward_ind.append(mean_train_num/count)
    #        mean_relative_reward_vec.append(mean_relative_reward/count)
    combined_rewards = []
    for i in range(len(dataManager_vec)):
        for j in range(len(relative_rewards_changed_vec[i])):
            if relative_rewards_changed_vec[i][j] != None:
                combined_rewards.append([dataManager_vec[i].train_num[j],relative_rewards_changed_vec[i][j]])
    combined_rewards = np.array(sorted(combined_rewards, key=lambda x: x[0]))

    combined_abs_rewards = []
    for i in range(len(dataManager_vec)):
        for j in range(len(relative_rewards_changed_vec[i])):
            combined_abs_rewards.append([dataManager_vec[i].train_num[j],sum(dataManager_vec[i].paths[j][0])*0.2])
    combined_abs_rewards = np.array(sorted(combined_abs_rewards, key=lambda x: x[0]))

    plt.figure(1)
    for j in range(len(relative_rewards_changed_vec)):
        plt.scatter(np.array(dataManager_vec[j].train_num),relative_rewards_changed_vec[j],marker = shape,alpha = 0.5)
    plt.plot(combined_rewards[:,0],lib.running_average(combined_rewards[:,1],5))
    plt.figure(2)
    for j in range(len(relative_rewards_changed_vec)):
        abs_rewards = [sum(dataManager_vec[j].paths[k][0])*0.2 for k in range(len(dataManager_vec[j].paths))]
        plt.scatter(np.array(dataManager_vec[j].train_num),abs_rewards,marker = shape,alpha = 0.5)
    plt.plot(combined_abs_rewards[:,0],lib.running_average(combined_abs_rewards[:,1],20))

#names = ["final_conv_new_reward_1","final_conv_new_reward_2","final_conv_new_reward_3","final_conv_new_reward_4"]
names = ["final_conv_new_reward_same_1"]
plot_rewards(names,'o')
#names = ["final_conv_analytic_new_reward_same_1"]#,"final_conv_analytic_new_reward_same_3","final_conv_analytic_new_reward_same_5"]
names = ["final_conv_analytic_new_reward_2","final_conv_analytic_new_reward_4","final_conv_analytic_new_reward_6","final_conv_analytic_new_reward_8"]#,"
plot_rewards(names,'x')
plt.show()
