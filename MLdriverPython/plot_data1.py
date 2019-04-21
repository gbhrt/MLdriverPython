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
    return sum(filtered_reward)/len(filtered_reward)

def get_data(folder,names_vec):
    HP = HyperParameters()
    train_indexes = [HP.save_every_train_number*j for j in range(1,101)]

    rewards_vec = []
    fails_vec = []
    for names in names_vec:#for every series of data (i.e. REVO or REVO+A)
        rewards = []
        fails = []
        for name in names:#for every training process
            single_rewards = []#average reward at each point
            single_fails = []
            for i in train_indexes:#for every test at a fixed parameter set (fixed training point)
                restore_path = os.getcwd()+ "\\files\\models\\"+str(folder)+"\\"+name+"\\"
                dataManager = data_manager1.DataManager(restore_path,restore_path,True,restore_name = 'data_manager_'+str(i))
                if dataManager.error:
                    print("cannot restore dataManager")
                    break
                single_rewards.append(get_avg_reward(dataManager))
                single_fails.append(get_avg_fails(dataManager))
            rewards.append(single_rewards)
            fails.append(single_fails)
        rewards_vec.append(rewards)
        fails_vec.append(fails)

    return train_indexes,rewards_vec,fails_vec

def plot_rewards(folder,names,shape=None,color=None,label = None,vod_label = None,max_train_iterations = 1000000):
    N = 1#80
    episodes_num = 50
    dataManager_vec = []
    relative_rewards_changed_vec = []
    for i in range(len(names)):
        HP.restore_name = names[i]
        HP.save_name = names[i]
        
        #low_friction_high_com_soft
        #slip_com_high
        #no_friction_hard
        
        save_path = os.getcwd()+ "\\files\\models\\"+str(folder)+"\\"+HP.save_name+"\\"
        restore_path = os.getcwd()+ "\\files\\models\\"+str(folder)+"\\"+HP.restore_name+"\\"
        dataManager_vec.append(data_manager1.DataManager(save_path,restore_path,True))
        relative_rewards_changed_vec.append(np.array(change_failes_value(dataManager_vec[-1].relative_reward,dataManager_vec[-1].episode_end_mode)))#[:episodes_num]

    max_len_ind = 0
    max_len = 0
    for i in range(len(dataManager_vec)):
        if dataManager_vec[i].train_num[-1] > max_len:
            max_len = dataManager_vec[i].train_num[-1]
            max_len_ind = i
    analytic_path = lib.compute_analytic_path(1111)
    max_tim_ind = 0
    for j,tim in enumerate (analytic_path.analytic_time):
        max_tim_ind = j
        if tim > 20:
            break




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
 

    #relative_reward_vec = []
    #for dataManager in dataManager_vec:
    #    relative_reward = []
    #    for j in range(len(dataManager.path_seed)):
    #        real_dis = dataManager.paths[j][1][-1]
    #        real_time = dataManager.paths[j][0][-1]
    #        analytic_path = lib.compute_analytic_path(dataManager.path_seed[j])
    #        for t,time in enumerate(analytic_path.analytic_time):
    #            if time >= real_time:
    #                break
    #        analytic_distance = analytic_path.distance[t]
    #        relative_reward.append(real_dis/analytic_distance)
    #    relative_reward_vec.append(relative_reward)

    combined_rewards = []
    for i in range(len(dataManager_vec)):
        for j in range(len(relative_rewards_changed_vec[i])):
            if relative_rewards_changed_vec[i][j] != None:
                combined_rewards.append([dataManager_vec[i].train_num[j],relative_rewards_changed_vec[i][j]])
    combined_rewards = np.array(sorted(combined_rewards, key=lambda x: x[0]))
    #combined_rewards = []
    #for i in range(len(dataManager_vec)):
    #    for j in range(len(relative_reward_vec[i])):
    #        combined_rewards.append([dataManager_vec[i].train_num[j],relative_reward_vec[i][j]])
    #combined_rewards = np.array(sorted(combined_rewards, key=lambda x: x[0]))

    fails_vec = []
    for i in range(len(dataManager_vec)):
        episode_fails = []
        for j in range(len(dataManager_vec[i].episode_end_mode)):
            if dataManager_vec[i].episode_end_mode[j] == 'kipp' or dataManager_vec[i].episode_end_mode[j] == 'deviate':
                episode_fails.append(1)
            else:
                episode_fails.append(0)
        fails_vec.append(episode_fails)

    for i in range(len(dataManager_vec)):
        fails = fails_vec[i].count(1)
        print("fail:",  fails/len(dataManager_vec[i].episode_end_mode)*100,"[%]", "lenght:",len(dataManager_vec[i].episode_end_mode),"check episodes")#
        
    fails_density_vec = []

    for i in range(len(fails_vec)):
        fails_density = lib.running_average(fails_vec[i],N)
        fails_density_vec.append(fails_density)

    combined_fails_density = []
    for i in range(len(dataManager_vec)):
        for j in range(len(fails_density_vec[i])):
            combined_fails_density.append([dataManager_vec[i].train_num[j],fails_density_vec[i][j]])
    combined_fails_density = np.array(sorted(combined_fails_density, key=lambda x: x[0]))


    ##combined_abs_rewards = []
    ##for i in range(len(dataManager_vec)):
    ##    for j in range(len(relative_rewards_changed_vec[i])):
    ##        #combined_abs_rewards.append([dataManager_vec[i].train_num[j],sum(dataManager_vec[i].paths[j][0])*0.2])
    ##        combined_abs_rewards.append([dataManager_vec[i].train_num[j],dataManager_vec[i].paths[j][1][-1]])
    ##combined_abs_rewards = np.array(sorted(combined_abs_rewards, key=lambda x: x[0]))





    plt.figure(1)
    for j in range(len(relative_rewards_changed_vec)):
        max_train_num = len(dataManager_vec[j].train_num)
        for i,train_n in enumerate (dataManager_vec[j].train_num):
            if train_n > max_train_iterations:
                max_train_num = i
                break
        #plt.scatter(np.array(dataManager_vec[j].train_num)[:max_train_num],(relative_rewards_changed_vec[j])[:max_train_num],marker = shape,alpha = 0.2,c = color)
    
    max_train_num = len(combined_rewards[:,0])
    for dataManager in dataManager_vec:
        for j,train_n in enumerate (combined_rewards[:,0]):
            if train_n > max_train_iterations:
                max_train_num = j
                break    
    ave = lib.running_average(combined_rewards[:,1],N)[:max_train_num]
    #plt.plot((combined_rewards[:,0])[:len(ave)],ave,c = color,label = label)
    plt.plot(ave,c = color,label = label)

    max_train_num = len(dataManager_vec[0].train_num)
    for i,train_n in enumerate (dataManager_vec[0].train_num):
        if train_n > max_train_iterations:
            max_train_num = i
            break
    ones = np.ones([len(dataManager_vec[max_len_ind].train_num)])[:max_train_num]
    
    #plt.plot(dataManager_vec[max_len_ind].train_num[:max_train_num],ones,linewidth = 2.0,c = 'r',label = vod_label)
    plt.plot(ones,linewidth = 2.0,c = 'r',label = vod_label)

    plt.legend()
    #plt.figure(1)
    #for j in range(len(relative_reward_vec)):
    #    plt.scatter(np.array(dataManager_vec[j].train_num)[:max_train_num],(relative_reward_vec[j])[:max_train_num],marker = shape,alpha = 0.2)#,c = color
    #ave = lib.running_average(combined_rewards[:,1],20)[:max_train_num]
    #plt.plot((combined_rewards[:,0])[:len(ave)],ave,c = color)

    #ones = np.ones([len(dataManager_vec[max_len_ind].train_num)])[:max_train_num]
    #plt.plot(dataManager_vec[max_len_ind].train_num[:max_train_num],ones,linewidth = 2.0,c = 'r')

    ##plt.figure(2)
    ##for j in range(len(relative_rewards_changed_vec)):
    ##    abs_rewards = [dataManager_vec[j].paths[k][1][-1] for k in range(len(dataManager_vec[j].paths))]
    ##    abs_rewards1 = [sum(dataManager_vec[j].paths[k][0])*0.2 for k in range(len(dataManager_vec[j].paths))]#for debugging
    ##    #for w in range(len(abs_rewards)):
    ##        #print("sum vel:",abs_rewards1[w],"dis:",abs_rewards[w])
    ##        #print(dataManager_vec[j].paths[w][1])

    ##    abs_rewards = abs_rewards[:max_train_num]
    ##    plt.scatter(np.array(dataManager_vec[j].train_num)[:max_train_num],abs_rewards,marker = shape,alpha = 0.2,c = color)
    ##ave = lib.running_average(combined_abs_rewards[:,1],N)[:max_train_num]
    ##plt.plot((combined_abs_rewards[:,0])[:len(ave)],ave,c = color,label = label)

    ##analytic_dist_vec = [analytic_path.distance[max_tim_ind] for _ in range(len(dataManager_vec[max_len_ind].train_num))]
    ##analytic_dist_vec = analytic_dist_vec[:max_train_num]
    ##plt.plot(dataManager_vec[max_len_ind].train_num[:max_train_num],analytic_dist_vec,linewidth = 2.0,c = 'r')
    ##plt.legend()

    max_train_num = len(combined_fails_density[:,0])
    for dataManager in dataManager_vec:
        for j,train_n in enumerate (combined_fails_density[:,0]):
            if train_n > max_train_iterations:
                max_train_num = j
                break
    #max_train_num =120#temp
    plt.figure(3)
    #for i in range(len(fails_density_vec)):
    #    plt.plot(np.array(dataManager_vec[i].train_num)[:len(fails_density_vec[i])],(fails_density_vec[i]))
    ave = lib.running_average(combined_fails_density[:,1],N)[:max_train_num]
    #plt.plot((combined_fails_density[:,0])[:len(ave)],ave*100,c = color,label = label)
    plt.plot(ave*100,c = color,label = label)
    plt.legend()

if __name__ == "__main__":
    folder = "paper_fix"
    names_vec = []
    names_vec.append(['REVO4'])#+str(i) for i in range(1,5)])
    #names_vec.append(['REVO+A'+str(i) for i in range(1,5)])
    save_every_n = 1
    trains,rewards_vec,fails_vec = get_data(folder,names_vec)
    size = 15
    plt.figure(1)
    plt.xlabel('Train iterations number',fontsize = size)
    plt.ylabel('Relative progress',fontsize = size)
    for names in names_vec:
        for rewards in rewards_vec:
            for single_rewards in rewards:
                plt.plot(trains[:len(single_rewards)],single_rewards)

    plt.figure(2)
    plt.xlabel('Train iterations number',fontsize = size)
    plt.ylabel('Fails [%]',fontsize = size)
    for names in fails_vec:
        for fails in fails_vec:
            for single_fails in fails:
                plt.plot(trains[:len(single_fails)],single_fails)


    plt.show()
