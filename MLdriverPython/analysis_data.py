import matplotlib.pyplot as plt
import numpy as np
import data_manager1
from hyper_parameters import HyperParameters
import os

import environment1
import classes
import library as lib
    



def correct_rewards(name):
    HP = HyperParameters()
    envData = environment1.OptimalVelocityPlannerData(env_mode = "DDPG")
    HP.restore_name = name
    HP.save_name = name
    save_path = os.getcwd()+ "\\files\\models\\final1\\"+HP.save_name+"\\"
    restore_path = os.getcwd()+ "\\files\\models\\final1\\"+HP.restore_name+"\\"
    dataManager = (data_manager1.DataManager(save_path,restore_path,True))

    #for i in range(len(dataManager.paths)):
    #    dataManager.paths[i] = [dataManager.paths[i],[]]
    for i in range(len(dataManager.paths)):
        dataManager.paths[i] = dataManager.paths[i][0]

    #relative_rewards = []
    #for i in range(len(dataManager.run_num)):
    #    if dataManager.episode_end_mode[i] != 'kipp' and dataManager.episode_end_mode[i] != 'deviate':
    #        real_mean_reward = dataManager.rewards[i]/envData.max_episode_steps *envData.max_velocity
    #    else:
    #        real_mean_reward = None
    #    path = classes.Path()
    #    path.position = lib.create_random_path(9000,0.05,seed = dataManager.path_seed[i])
    #    lib.comp_velocity_limit_and_velocity(path,skip = 10,reduce_factor = 1.0)
    #    max_time_ind = 0
    #    for i,tim in enumerate (path.analytic_time):
    #        max_time_ind = i
    #        if tim > envData.max_episode_steps*envData.step_time:
    #            break
    #    analytic_mean_vel = path.distance[max_time_ind] / (envData.max_episode_steps*envData.step_time)
    #    if real_mean_reward != None:
    #        relative_rewards.append(real_mean_reward/analytic_mean_vel)
    #    else:
    #         relative_rewards.append(None)


    ##plt.scatter(dataManager.run_num,relative_rewards)
    ##plt.scatter(dataManager.run_num,dataManager.relative_reward)
    ##plt.show()
    #dataManager.relative_reward = relative_rewards
    dataManager.save_data()



if __name__ == "__main__": 
   # names = ["final_conv_analytic_new_reward_2","final_conv_analytic_new_reward_4","final_conv_analytic_new_reward_6","final_conv_analytic_new_reward_6"]
   names =["final_conv_analytic_new_reward_6"] 
   for name in names:
        correct_rewards(name)

