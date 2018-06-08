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
        save_path = os.getcwd()+ "\\files\\models\\final\\"+HP.save_name+"\\"
        restore_path = os.getcwd()+ "\\files\\models\\final\\"+HP.restore_name+"\\"
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
        ave = lib.running_average(combined_rewards[:,1],20)
    plt.plot((combined_rewards[:,0])[:len(ave)],ave)
    plt.figure(2)
    for j in range(len(relative_rewards_changed_vec)):
        abs_rewards = [sum(dataManager_vec[j].paths[k][0])*0.2 for k in range(len(dataManager_vec[j].paths))]
        plt.scatter(np.array(dataManager_vec[j].train_num),abs_rewards,marker = shape,alpha = 0.5)
    ave = lib.running_average(combined_abs_rewards[:,1],20)
    plt.plot((combined_abs_rewards[:,0])[:len(ave)],ave)

    analytic_path = lib.compute_analytic_path(1111)
    max_tim_ind = 0
    for j,tim in enumerate (analytic_path.analytic_time):
        max_tim_ind = j
        if tim > 20:
            break
    max_len_ind = 0
    max_len = 0
    for i in range(len(dataManager_vec)):
        if dataManager_vec[i].train_num[-1] > max_len:
            max_len = dataManager_vec[i].train_num[-1]
            max_len_ind = i
    analytic_dist_vec = [analytic_path.distance[max_tim_ind] for _ in range(len(dataManager_vec[0].train_num))]
    plt.plot(dataManager_vec[max_len_ind].train_num,analytic_dist_vec,linewidth = 2.0)

if __name__ == "__main__":
    #convolution:
    #names1 = ["final_conv_analytic_new_reward_2","final_conv_analytic_new_reward_4","final_conv_analytic_new_reward_6","final_conv_analytic_new_reward_8"]#,"
    #names2 = ["final_conv_new_reward_1","final_conv_new_reward_2","final_conv_new_reward_3","final_conv_new_reward_4"]
    #names1 = ["final_conv_new_reward_same_1"]
    #names2 = ["final_conv_analytic_new_reward_same_1"]#,"final_conv_analytic_new_reward_same_3","final_conv_analytic_new_reward_same_5"]

    #same path for testing. 2 trains per step time:
    #names1 = ["final_analytic_2","final_analytic_4","final_analytic_6","final_analytic_8","final_analytic_10"]
    names1 = ["‏‏final_analytic_1_1"]
    names2 = ["final_1","final_3","final_5","final_7","final_9"]


    #same path (seed = 1111) for testing. different training steps per step time:
    #names1 = ["final_analytic_1_4","final_analytic_3_4"]
    #names2 = ["final_11_4","final_13_4_short","final_15_4","final_17_4","final_19_4"]

    #random path for testing: different training steps per step time:
    #names1 = ["final_analytic_2_10","final_analytic_4_10","final_analytic_6_10","final_analytic_8_10","final_analytic_10_10"]#,
    #names2 = ["final_4_10","final_6_10"]#,"final_1
    plot_rewards(names1,'o')
    plot_rewards(names2,'x')
    plt.show()
