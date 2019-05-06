import matplotlib
matplotlib.rcParams.update({'errorbar.capsize': 2})
import matplotlib.pyplot as plt
import numpy as np
import data_manager1
from hyper_parameters import HyperParameters
import os
import library as lib
import copy
import classes

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
    
def get_relative_reward(dataManager):
    run_time =  0.2*101
    relative_reward = []
    for real_path,seed,end_mode in zip(dataManager.paths,dataManager.path_seed, dataManager.episode_end_mode):
        if end_mode == 'kipp' or end_mode == 'deviate' or len(real_path[1]) == 0:
            relative_reward.append(-1)
        else:
            real_dist = real_path[1][100]
            path = classes.Path()
            path.position = lib.create_random_path(6000,0.05,seed = seed)#9000
            lib.comp_velocity_limit_and_velocity(path,skip = 10,reduce_factor = 1.0)
            for i,tim in enumerate (path.analytic_time):
                if tim > run_time:
                    break
            if i == len(path.analytic_time) - 1:
                print("create longer path")
                path.position = lib.create_random_path(10000,0.05,seed = seed)#9000
                lib.comp_velocity_limit_and_velocity(path,skip = 10,reduce_factor = 1.0)
                for i,tim in enumerate (path.analytic_time):
                    if tim > run_time:
                        break

            analytic_dist = path.distance[i]*1.06
        

            relative_reward.append(real_dist/analytic_dist)
    return relative_reward

def get_avg_reward(dataManager):
    filtered_reward = [value for value in dataManager.relative_reward if value != -1]
    if len(filtered_reward)>0:
        return sum(filtered_reward)/len(filtered_reward)
    return -1

def correct_relative_reward(folder,names_vec):
    HP = HyperParameters()
    train_indexes = [5000*j for j in range(1,50)]
    for names in names_vec:#for every series of data (e.g. REVO or REVO+A)
        for name in names[0]:#for every training process(e.g. REVO1)
            for i in train_indexes:#for every test at a fixed parameter set (fixed training point)
                restore_path = os.getcwd()+ "\\files\\models\\"+str(folder)+"\\"+name+"\\"
                restore_name = 'data_manager_'+str(i)
                dataManager = data_manager1.DataManager(restore_path,restore_path,True,save_name = restore_name,restore_name = restore_name)#
                if dataManager.error:
                    print("cannot restore dataManager")
                    break
                dataManager.relative_reward  = get_relative_reward(dataManager) 
                dataManager.save_data()
                del dataManager

    return 

def add_zero_data_manager(folder,names_vec):
    for names in names_vec:#for every series of data (e.g. REVO or REVO+A)
        for name in names[0]:#for every training process(e.g. REVO1)
            save_path = os.getcwd()+ "\\files\\models\\"+str(folder)+"\\"+name+"\\"
            save_name = 'data_manager_'+str(0)
            dataManager = data_manager1.DataManager(save_path,save_path,False,save_name = save_name)#
            dataManager.episode_end_mode = ['max steps' for _ in range(101)]
            dataManager.relative_reward = [0.0 for _ in range(101)]
            dataManager.save_data()


def get_data(folder,names_vec):
    HP = HyperParameters()

    train_indexes = [5000*j for j in range(0,20)]
    rewards_vec = []
    reward_vec_indexes = []
    fails_vec = []
    series_colors = []
    series_names = []
    for names in names_vec:#for every series of data (e.g. REVO or REVO+A)
        rewards = []
        reward_indexes = []
        fails = []
        series_colors.append(names[1][1])
        series_names.append(names[1][0])
        for name in names[0]:#for every training process(e.g. REVO1)
          
            single_rewards = []#average reward at each point
            single_reward_indexes = []
            single_fails = []
            for i in train_indexes:#for every test at a fixed parameter set (fixed training point)
                restore_path = os.getcwd()+ "\\files\\models\\"+str(folder)+"\\"+name+"\\"
                #if name ==  "VOD_01_1" or name == "VOD_022_1" or name == "VOD_020_1" or name == "VOD_010_1":
                if name in(["VOD_00","VOD_002","VOD_004","VOD_006","VOD_008","VOD_01","VOD_012","VOD_014","VOD_016","VOD_018","VOD_02","VOD_015","VOD_005","VOD_0175"]):
                    restore_name = 'data_manager_0'
                else:
                    restore_name = 'data_manager_'+str(i)
                dataManager = data_manager1.DataManager(restore_path,restore_path,True,restore_name = restore_name)#
                if dataManager.error:
                    print("cannot restore dataManager",name,"num:",i)
                    break
                avg_reward = get_avg_reward(dataManager)
                if avg_reward>=0:
                    single_rewards.append(avg_reward)
                    single_reward_indexes.append(i)
                single_fails.append(get_avg_fails(dataManager))
            rewards.append(single_rewards)
            reward_indexes.append(single_reward_indexes)
            fails.append(single_fails)
        rewards_vec.append(rewards)
        reward_vec_indexes.append(reward_indexes)
        fails_vec.append(fails)

    return reward_vec_indexes,rewards_vec,train_indexes,fails_vec,series_colors,series_names

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
    folder = "paper_fix"#\old reward backup"
    size = 15
    names_vec = []

    #names = ["VOD_00","VOD_002","VOD_004","VOD_006","VOD_008","VOD_01"]
    #names = ["VOD_00","VOD_005","VOD_01","VOD_015","VOD_0175","VOD_02"]
    #for name in names:
    #    names_vec.append([[name],[name,None]])
    #reward_vec_indexes,rewards_vec,indexes,fails_vec,series_colors,series_names = get_data(folder,names_vec)
    #vel = [reward[0][0] for reward in rewards_vec]
    #fail = [fails[0][0] for fails in fails_vec]
    
    #vel = [vel[i] / vel[0] for i in range(len(vel))]
    #plt.figure(3)
    #plt.xlabel('Velocity factor',fontsize = size)
    #plt.ylabel('Fails [%]',fontsize = size)
    #plt.plot(vel,fail,'o',c = 'r')
    #plt.plot(vel,fail,c = 'r')
    #names_vec.append(['same_REVO2'])#,'same_REVO2'])#+str(i) for i in range(1,5)])
    #names_vec.append(['eps_REVO1','eps_REVO2','eps_REVO3'])
    #names_vec.append([['batch_256_REVO1','batch_256_REVO2','batch_256_REVO3','batch_256_REVO4','batch_256_REVO5'],['batch_256_REVO','blue']])

    #names_vec.append([['copyREVO+FA3'],['REVO+FA','orange']])

    names_vec.append([['REVO6','REVO7','REVO8','REVO9','REVO10'],['REVO','green']])#corrected
    names_vec.append([['REVO+A1','REVO+A2','REVO+A3','REVO+A4','REVO+A6'],['REVO+A','black']])#corrected ,
    names_vec.append([['REVO+F1','REVO+F2','REVO+F3','REVO+F4','REVO+F5'],['REVO+F','blue']])#corrected,'REVO+F5' ,'REVO+F2','REVO+F3','REVO+F4' REVO+F1, data_manager_15000 was empty, replaced by 20000
    names_vec.append([['REVO+FA1','REVO+FA2' ,'REVO+FA3','REVO+FA4','REVO+FA5'],['REVO+FA','orange']])#corrected   'REVO+FA1' ,'REVO+FA3','REVO+FA4'
    #names_vec.append([['REVO+FA1'],['REVO+FA1',None]])
    #names_vec.append([['REVO+FA2'],['REVO+FA2',None]])
    #names_vec.append([['REVO+FA3'],['REVO+FA3',None]])
    #names_vec.append([['REVO+FA4'],['REVO+FA4',None]])
    #names_vec.append([['REVO+FA5'],['REVO+FA5',None]])
    #names_vec.append([['REVO+FA6'],['REVO+FA6',None]])
    #names_vec.append([['REVO+FA7'],['REVO+FA7',None]])
    #names_vec.append([['REVO+FA8'],['REVO+FA8',None]])
    #names_vec.append([['REVO+FA9'],['REVO+FA9',None]])
    #names_vec.append([['REVO+FA6','REVO+FA7','REVO+FA8','REVO+FA9'],['REVO+FA','blue']])
    #names_vec.append([["same_REVO3"],['REVO','green']])#"same_REVO1" "same_REVO2",,"same_REVO5" ,"same_REVO4"
    #names_vec.append([["same_REVO+F5",],['same_REVO+F1','blue']])
    
    #add_zero_data_manager(folder,names_vec)
    #correct_relative_reward(folder,names_vec)

    reward_vec_indexes,rewards_vec,indexes,fails_vec,series_colors,series_names = get_data(folder,names_vec)

    avg_rewards_vec,var_rewards_vec = average_training_processes(rewards_vec)
    avg_fails_vec,var_fails_vec = average_training_processes(fails_vec)
   

    
    plt.figure(1)
    
    plt.xlabel('Train iterations number',fontsize = size)
    plt.ylabel('Relative progress',fontsize = size)
    #for rewards,color in zip(rewards_vec,series_colors):
    #    for single_rewards in rewards:
    #        plt.scatter(trains[:len(single_rewards)],single_rewards,c = color,alpha = 0.5)#
    plt.xlim(0,100000)
    for reward_indexes,rewards,var,color,label in zip(reward_vec_indexes,avg_rewards_vec,var_rewards_vec,series_colors,series_names):
        plt.plot(indexes[:len(rewards)],rewards,'o',color = color,label = label)
        plt.errorbar(indexes[:len(rewards)],rewards,var,c = color,alpha = 0.7)#reward_indexes[0]

    plt.plot([0,indexes[len(rewards)-1]],[1,1],linewidth = 2.0,c = 'r',label = 'VOD')
    plt.legend()

    plt.figure(2)
    plt.xlim(0,100000)
    plt.ylim(0,100)
    plt.xlabel('Train iterations number',fontsize = size)
    plt.ylabel('Fails [%]',fontsize = size)
    #for fails,color in zip(fails_vec,series_colors):
    #    for single_fails in fails:
    #        plt.scatter(indexes[:len(single_fails)],single_fails,c = color,alpha = 0.5)
    for fails,var,color,label in zip(avg_fails_vec,var_fails_vec,series_colors,series_names):
        plt.plot(indexes[:len(fails)],fails,color = color,label = label)
        #plt.errorbar(indexes[:len(fails)],fails,var,c = color,alpha = 0.7)
    plt.legend()




    plt.show()
