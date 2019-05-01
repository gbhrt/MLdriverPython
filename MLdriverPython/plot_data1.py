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

def get_avg_fails(dataManager):
    return (dataManager.episode_end_mode.count('kipp')+dataManager.episode_end_mode.count('deviate'))*100/len(dataManager.episode_end_mode)
    
def get_avg_reward(dataManager):
    filtered_reward = [value for value in dataManager.relative_reward if value != 0]
    if len(filtered_reward)>0:
        return sum(filtered_reward)/len(filtered_reward)
    return 0.0

def get_data(folder,names_vec):
    HP = HyperParameters()

    train_indexes = [5000*j for j in range(1,20)]
    rewards_vec = []
    fails_vec = []
    series_colors = []
    series_names = []
    for names in names_vec:#for every series of data (e.g. REVO or REVO+A)
        rewards = []
        fails = []
        series_colors.append(names[1][1])
        series_names.append(names[1][0])
        for name in names[0]:#for every training process(e.g. REVO1)
          
            single_rewards = []#average reward at each point
            single_fails = []
            for i in train_indexes:#for every test at a fixed parameter set (fixed training point)
                restore_path = os.getcwd()+ "\\files\\models\\"+str(folder)+"\\"+name+"\\"
                if name ==  "VOD_01_1" or name == "VOD_022_1" or name == "VOD_020_1" or name == "VOD_010_1":
                    restore_name = 'data_manager_0'
                else:
                    restore_name = 'data_manager_'+str(i)
                dataManager = data_manager1.DataManager(restore_path,restore_path,True,restore_name = restore_name)#
                if dataManager.error:
                    print("cannot restore dataManager")
                    break
                single_rewards.append(get_avg_reward(dataManager))
                single_fails.append(get_avg_fails(dataManager))
            rewards.append(single_rewards)
            fails.append(single_fails)
        rewards_vec.append(rewards)
        fails_vec.append(fails)

    return train_indexes,rewards_vec,fails_vec,series_colors,series_names

def average_training_processes(rewards_vec):
    avg_rewards_vec = []
    vars_vec = []
    for rewards in rewards_vec:
        rewards_avg = []
        vars = []
        for i in range(len(rewards[0])):#colums num
            s = 0
            row = []
            try:
                
                for j in range(len(rewards)):#rows
                    row.append(rewards[j][i])
                var = np.sqrt(np.var(row))
                s = sum(row)
            except:
                break
            rewards_avg.append(s/len(rewards))
            vars.append(var)
        vars_vec.append(vars)
        avg_rewards_vec.append(rewards_avg)  
    return avg_rewards_vec,vars_vec

if __name__ == "__main__":
    folder = "paper_fix"
    names_vec = []
    #names_vec.append(['same_REVO2'])#,'same_REVO2'])#+str(i) for i in range(1,5)])
    #names_vec.append(['eps_REVO1','eps_REVO2','eps_REVO3'])

    #names_vec.append([['REVO6','REVO7','REVO8','REVO9'],['REVO','green']])#'REVO1','REVO4'#['REVO2','REVO3','REVO5']
    #names_vec.append([['REVO+A1','REVO+A2','REVO+A3','REVO+A4'],['REVO+A','black']])
    #names_vec.append([['REVO+F1','REVO+F2','REVO+F3','REVO+F4'],['REVO+F','blue']])#,'REVO+F5' ,'REVO+F2','REVO+F3','REVO+F4' REVO+F1, data_manager_15000 was empty, replaced by 20000
    #names_vec.append([['REVO+FA1','REVO+FA2','REVO+FA3','REVO+FA4'],['REVO+FA','orange']])

    names_vec.append([['VOD_01_1'],['REVO+VOD_01_1','orange']])
    names_vec.append([['VOD_022_1'],['REVO+VOD_022_1','black']])
    names_vec.append([['VOD_020_1'],['REVO+VOD_020_1','green']])
    names_vec.append([['VOD_010_1'],['REVO+VOD_010_1','blue']])
    
    trains,rewards_vec,fails_vec,series_colors,series_names = get_data(folder,names_vec)


    avg_rewards_vec,var_rewards_vec = average_training_processes(rewards_vec)
    avg_fails_vec,var_fails_vec = average_training_processes(fails_vec)
   

    size = 15
    plt.figure(1)
    
    plt.xlabel('Train iterations number',fontsize = size)
    plt.ylabel('Relative progress',fontsize = size)
    #for rewards,color in zip(rewards_vec,series_colors):
    #    for single_rewards in rewards:
    #        plt.scatter(trains[:len(single_rewards)],single_rewards,c = color,alpha = 0.5)#
    for rewards,var,color,label in zip(avg_rewards_vec,var_rewards_vec,series_colors,series_names):
        plt.plot(trains[:len(rewards)],rewards,'o',color = color,label = label)
        plt.errorbar(trains[:len(rewards)],rewards,var,c = color,alpha = 0.7)

    plt.plot([0,trains[len(rewards)-1]],[1,1],linewidth = 2.0,c = 'r',label = 'VOD')
    plt.legend()

    plt.figure(2)
    plt.xlabel('Train iterations number',fontsize = size)
    plt.ylabel('Fails [%]',fontsize = size)
    #for fails,color in zip(fails_vec,series_colors):
    #    for single_fails in fails:
    #        plt.scatter(trains[:len(single_fails)],single_fails,c = color,alpha = 0.5)
    for fails,var,color,label in zip(avg_fails_vec,var_fails_vec,series_colors,series_names):
        plt.plot(trains[:len(fails)],fails,'o',color = color,label = label)
        plt.errorbar(trains[:len(fails)],fails,var,c = color,alpha = 0.7)
    plt.legend()




    plt.show()
