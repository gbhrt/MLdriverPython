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
def get_avg_violation_count(dataManager):
    
    return 100*sum(dataManager.violation_count)/sum(dataManager.length)

def get_abs_vel(dataManager):
    run_time =  0.2*101
    abs_vel_vec = []
    var_vec = []
    for real_path,seed,end_mode in zip(dataManager.paths,dataManager.path_seed, dataManager.episode_end_mode):
        if end_mode == 'kipp' or end_mode == 'deviate' or len(real_path[1]) == 0:
            continue
        else:
            vel = sum(real_path[0])/len(real_path[0])
            var = np.sqrt(np.var(real_path[0]))
            abs_vel_vec.append(vel)
            var_vec.append(var)
    abs_vel = sum(abs_vel_vec)/len(abs_vel_vec)
    var = sum(var_vec)/len(var_vec)
    return abs_vel,var

def get_relative_to_baseline_reward(dataManager,baseline_dist_vec,baseline_seeds,length):
    run_time =  0.2*length
    relative_reward = []
    for real_path,seed,end_mode,baseline_dist,baseline_seed in zip(dataManager.paths,dataManager.path_seed, dataManager.episode_end_mode,baseline_dist_vec,baseline_seeds):
        if baseline_seed!=seed:
            try:
                baseline_dist = baseline_dist_vec[baseline_seeds.index(seed)]
            except:
                print("baseline_seeds not contain:",seed)
        if end_mode == 'kipp' or end_mode == 'deviate' or len(real_path[1]) == 0:
            relative_reward.append(0)
        else:
            real_dist = real_path[1][length-1]
            if abs(baseline_dist) > 1e-6:
                relative_reward.append(real_dist/baseline_dist)
            else:
                print("error - baseline distance is 0")
                relative_reward.append(10000)
    return relative_reward
def get_relative_reward(dataManager):
    run_time =  0.2*100
    relative_reward = []
    for real_path,seed,end_mode in zip(dataManager.paths,dataManager.path_seed, dataManager.episode_end_mode):
        if end_mode == 'kipp' or end_mode == 'deviate' or len(real_path[1]) == 0:
            relative_reward.append(-1)
        else:
            real_dist = real_path[1][99]
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
    #filtered_reward = [value for value in dataManager.relative_reward if value != -1]
    filtered_reward = [value for value in dataManager.relative_reward if value != 0]
    if len(filtered_reward)>0:
        return sum(filtered_reward)/len(filtered_reward),np.std(filtered_reward)
    return -1,0

def get_baseline_dist(folder,baseline,length):
    restore_path = os.getcwd()+ "/files/models/"+str(folder)+"/"+baseline+"/"
    restore_name = 'data_manager'
    baseline_dist = []
    dataManager = data_manager1.DataManager(restore_path,restore_path,True,save_name = restore_name,restore_name = restore_name)#
    print("seeds:",dataManager.path_seed)
    for real_path,seed,end_mode in zip(dataManager.paths,dataManager.path_seed, dataManager.episode_end_mode):
        if end_mode == 'kipp' or end_mode == 'deviate' or len(real_path[1]) == 0:
            print('error - baseline failed')
        else:
            baseline_dist.append(real_path[1][length-1])
    
    return baseline_dist,dataManager.path_seed

#def correct_relative_reward(folder,names_vec,train_indexes,baseline):
#    length = 100
#    #HP = HyperParameters()
#    baseline_dist = get_baseline_dist(folder,baseline,length)
    
#    for names in names_vec:#for every series of data (e.g. REVO or REVO+A)
#        for name in names[0]:#for every training process(e.g. REVO1)
#            for i in train_indexes:#for every test at a fixed parameter set (fixed training point)
#                restore_path = os.getcwd()+ "/files/models/"+str(folder)+"/"+name+"/"
#                restore_name = 'data_manager_'+str(i)
#                dataManager = data_manager1.DataManager(restore_path,restore_path,True,save_name = restore_name,restore_name = restore_name)#
#                if dataManager.error:
#                    print("cannot restore dataManager")
#                    continue
#                dataManager.relative_reward  = get_relative_reward(dataManager) 
#                #dataManager.relative_reward  = get_relative_reward_real_vod(dataManager,VOD_dist)
#                dataManager.save_data()
#                del dataManager
#    return 
def comp_relative_reward(folder,names_vec,train_indexes,baseline):
    length = 100
    baseline_dist,baseline_seeds = get_baseline_dist(folder,baseline,length)
    print("baseline_dist:",baseline_dist)
    for names in names_vec:#for every series of data (e.g. REVO or REVO+A)
        for name in names[0]:#for every training process(e.g. REVO1)
            for i in train_indexes:#for every test at a fixed parameter set (fixed training point)
                restore_path = os.getcwd()+ "/files/models/"+str(folder)+"/"+name+"/"
                restore_name = 'data_manager_'+str(i)
                dataManager = data_manager1.DataManager(restore_path,restore_path,True,save_name = restore_name,restore_name = restore_name)#
                if dataManager.error:
                    print("cannot restore dataManager")
                    continue
                print("seeds:",dataManager.path_seed)
                #dataManager.relative_reward  = get_relative_reward(dataManager) 
                dataManager.relative_reward  = get_relative_to_baseline_reward(dataManager,baseline_dist,baseline_seeds,length)
                print("dataManager.relative_reward:",dataManager.relative_reward )
                dataManager.save_data()
                del dataManager
    return 

def add_zero_data_manager(folder,names_vec):
    for names in names_vec:#for every series of data (e.g. REVO or REVO+A)
        for name in names[0]:#for every training process(e.g. REVO1)
            save_path = os.getcwd()+ "/files/models/"+str(folder)+"/"+name+"/"
            save_name = 'data_manager_'+str(0)
            dataManager = data_manager1.DataManager(save_path,save_path,False,save_name = save_name)#
            dataManager.episode_end_mode = ['max steps' for _ in range(101)]
            dataManager.relative_reward = [0.0 for _ in range(101)]
            dataManager.save_data()


def get_training_process_data(folder,names_vec,train_indexes,baseline_folder,baseline_name):
    HP = HyperParameters()
    length = 100
    baseline_dist,baseline_seeds = get_baseline_dist(baseline_folder,baseline_name,length)
    rewards_vec = []
    var_vec =[]
    reward_vec_indexes = []
    fails_vec = []
    fails_var_vec = []
    violation_count_vec = []
    stabilize_count_brake_vec = []
    stabilize_count_steer_vec = []
    series_colors = []
    series_names = []
    for names in names_vec:#for every series of data (e.g. REVO or REVO+A)
        vec_learning_process_rewards = []#save all learning processes
        vec_learning_process_var = []
        vec_learning_process_reward_indexes = []
        vec_learning_process_fails = []
        vec_learning_process_violation_count = []
        vec_learning_process_episode_length = []
        vec_stabilize_count_brake = []
        vec_stabilize_count_steer = []
        for name in names[0]:#for every training process(e.g. REVO1)
            restore_path = os.getcwd()+ "/files/models/"+str(folder)+"/"+name+"/"
            restore_name = 'data_manager'
            print("name:",name)
            dataManager = data_manager1.DataManager(restore_path,restore_path,True,restore_name = restore_name,save_name = restore_name)#
            if dataManager.error:
                print("cannot restore dataManager",name)
                break
            vec_learning_process_rewards.append(get_relative_to_baseline_reward(dataManager,baseline_dist,baseline_seeds,length))
            vec_learning_process_violation_count.append(dataManager.violation_count)
            vec_learning_process_fails.append([1 if end_mode == 'kipp' or end_mode == 'deviate' else 0 for end_mode in dataManager.episode_end_mode])
            vec_learning_process_episode_length.append(dataManager.length)
            vec_learning_process_reward_indexes = dataManager.run_num
            vec_stabilize_count_brake.append(dataManager.stabilize_count_brake)
            vec_stabilize_count_steer.append(dataManager.stabilize_count_steer)


        tr_vec_learning_process_rewards = list(map(list, zip(*vec_learning_process_rewards)))#transpose
        tr_vec_learning_process_violation_count = list(map(list, zip(*vec_learning_process_violation_count)))#transpose
        tr_vec_learning_process_fails = list(map(list, zip(*vec_learning_process_fails)))#transpose
        tr_vec_learning_process_episode_length = list(map(list, zip(*vec_learning_process_episode_length)))#transpose
        tr_vec_stabilize_count_brake = list(map(list, zip(*vec_stabilize_count_brake)))#transpose
        tr_vec_stabilize_count_steer = list(map(list, zip(*vec_stabilize_count_steer)))#transpose


        rewards = []
        var =[]
        reward_indexes = []
        fails = []
        fails_var = []
        violation_count = []
        stabilize_count_brake = []
        stabilize_count_steer = []
        series_colors.append(names[1][1])
        series_names.append(names[1][0])
        for episode_learning_process_rewards,episode_learning_process_violation_count ,episode_learning_process_fail,episode_learning_process_episode_length,episode_stabilize_count_brake,episode_stabilize_count_steer\
            in zip(tr_vec_learning_process_rewards,tr_vec_learning_process_violation_count,tr_vec_learning_process_fails,tr_vec_learning_process_episode_length,tr_vec_stabilize_count_brake,tr_vec_stabilize_count_steer):
            sum_episode_lengths = sum(episode_learning_process_episode_length)
            filtered_reward = [episode_learning_process_reward for episode_learning_process_reward in episode_learning_process_rewards if episode_learning_process_reward != 0 ]
            rewards.append(sum(filtered_reward)/len(filtered_reward))
            var.append(np.std(episode_learning_process_rewards))
            violation_count.append(100*sum(episode_learning_process_violation_count)/sum_episode_lengths)
            fails.append(100*sum(episode_learning_process_fail)/len(episode_learning_process_episode_length))#sum_episode_lengths
            fails_var.append(np.std(episode_learning_process_fail).astype(float))
            stabilize_count_brake.append(100*sum(episode_stabilize_count_brake)/sum_episode_lengths)
            stabilize_count_steer.append(100*sum(episode_stabilize_count_steer)/sum_episode_lengths)

        rewards_vec.append(rewards)
        var_vec.append(var)
        reward_vec_indexes.append(vec_learning_process_reward_indexes)
        fails_vec.append(fails)
        violation_count_vec.append(violation_count)
        fails_var_vec.append(fails_var)
        stabilize_count_brake_vec.append(stabilize_count_brake)
        stabilize_count_steer_vec.append(stabilize_count_steer)
        

    return reward_vec_indexes,rewards_vec,reward_vec_indexes[0],fails_vec,var_vec,series_colors,series_names,violation_count_vec,fails_var_vec,stabilize_count_brake_vec,stabilize_count_steer_vec


def get_data(folder,names_vec,train_indexes, return_violation_count = False):#, take_0_flag = False
    HP = HyperParameters()

    rewards_vec = []
    var_vec =[]
    reward_vec_indexes = []
    fails_vec = []
    violation_count_vec = []
    series_colors = []
    series_names = []
    for names in names_vec:#for every series of data (e.g. REVO or REVO+A)
        rewards = []
        var =[]
        reward_indexes = []
        fails = []
        violation_count = []
        series_colors.append(names[1][1])
        series_names.append(names[1][0])
        for name in names[0]:#for every training process(e.g. REVO1)
          
            single_rewards = []#average reward at each point
            single_var = []
            single_reward_indexes = []
            single_fails = []
            single_violation_count = []
            for i in train_indexes:#for every test at a fixed parameter set (fixed training point)
                restore_path = os.getcwd()+ "/files/models/"+str(folder)+"/"+name+"/"
                restore_name = 'data_manager_'+str(i)#'data_manager'
                print("name:",name)
                dataManager = data_manager1.DataManager(restore_path,restore_path,True,restore_name = restore_name,save_name = restore_name)#
                if dataManager.error:
                    print("cannot restore dataManager",name,"num:",i)
                    break
                #dataManager.save_data()
                #print(dataManager.path_seed)
                avg_reward ,var_reward= get_avg_reward(dataManager)
                vel,abs_var = 0,0#get_abs_vel(dataManager)
                if avg_reward>=0:
                    single_rewards.append(avg_reward)
                    single_var.append(var_reward)
                    single_reward_indexes.append(i)
                single_fails.append(get_avg_fails(dataManager))
                if return_violation_count:
                    single_violation_count.append(get_avg_violation_count(dataManager))
                    print("violation_count:", single_violation_count[-1])
                print(name," ",i,"abs_vel:",vel,"var_abs_vel:",abs_var,"avg_reward:",avg_reward,"var_reward:",var_reward,"single_fails:",single_fails[-1],"len:",len(dataManager.path_seed))
            rewards.append(single_rewards)
            var.append(single_var)
            reward_indexes.append(single_reward_indexes)
            fails.append(single_fails)
            violation_count.append(single_violation_count)
        rewards_vec.append(rewards)
        var_vec.append(var)
        reward_vec_indexes.append(reward_indexes)
        fails_vec.append(fails)
        violation_count_vec.append(violation_count)
    if return_violation_count:
        return reward_vec_indexes,rewards_vec,train_indexes,fails_vec,var_vec,series_colors,series_names,violation_count_vec
    else:
        return reward_vec_indexes,rewards_vec,train_indexes,fails_vec,var_vec,series_colors,series_names

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

def plot_fails_vel_comparison():
    folder = "MB_paper_stabilize"#"MB_paper" model_based
    #names_vec.append([['MB_R_4'],['Model Based RL',None]])
    #names_vec.append([['MB_R_2'],['Model Based RL',None]])#,'MB_R_2'


    #var_vec_VOD_const = [0.01*i for i in range(0,20)]
    #names = ['VOD_var_check_const2_'+str(var) for var in var_vec_VOD_const]
    #names_vec = []
    #for name in names:
    #   names_vec.append([[name],[name,None]])
    #reward_vec_indexes,rewards_vec,indexes,fails_vec,var,series_colors,series_names,violation_count_vec = get_data(folder,names_vec,[0],return_violation_count = True)
    #vel_VOD_const = [reward[0][0] for reward in rewards_vec]
    #fail_VOD_const = [fails[0][0] for fails in fails_vec]
    #violation_count_VOD_const = [violation_counts[0][0] for violation_counts in violation_count_vec]

    var_vec_VOD_linear = [0.01*i for i in range(0,15)]
    #names = ['VOD_var_check_linear2_'+str(var) for var in var_vec_VOD_linear]#19
    #names = ['VOD_var_check_linear3_'+str(var) for var in var_vec_VOD_linear]#19
    names = ['MB_stabilize_'+str(var) for var in var_vec_VOD_linear]#19
    
    names_vec = []
    for name in names:
       names_vec.append([[name],[name,None]])
    reward_vec_indexes,rewards_vec,indexes,fails_vec,var,series_colors,series_names,violation_count_vec = get_data(folder,names_vec,[0],return_violation_count = True)
    vel_VOD_linear = [reward[0][0] for reward in rewards_vec]
    fail_VOD_linear = [fails[0][0] for fails in fails_vec]
    violation_count_VOD_linear = [violation_counts[0][0] for violation_counts in violation_count_vec]
    
    #folder = 'MB_paper_stabilize/MB_learning_process2'
    var_vec_MB_linear = [0.01*i for i in range(0,15)]
    #names = ['MB_var_check_linear2_'+str(var) for var in var_vec_MB_linear]#14
    names = ['MB_no_stabilize_'+str(var) for var in var_vec_MB_linear]#14
    names_vec = []
    for name in names:
       names_vec.append([[name],[name,None]])
    reward_vec_indexes,rewards_vec,indexes,fails_vec,var,series_colors,series_names,violation_count_vec = get_data(folder,names_vec,[0],return_violation_count = True)
    vel_MB_linear = [reward[0][0] for reward in rewards_vec]
    fail_MB_linear = [fails[0][0] for fails in fails_vec]
    violation_count_MB_linear = [violation_counts[0][0] for violation_counts in violation_count_vec]

    #var_vec_MB_const = [0.01*i for i in range(0,20)]
    #names = ['MB_var_check_const2_'+str(var) for var in var_vec_MB_const]#14
    #names_vec = []
    #for name in names:
    #   names_vec.append([[name],[name,None]])
    #reward_vec_indexes,rewards_vec,indexes,fails_vec,var,series_colors,series_names,violation_count_vec = get_data(folder,names_vec,[0],return_violation_count = True)
    #vel_MB_const = [reward[0][0] for reward in rewards_vec]
    #fail_MB_const = [fails[0][0] for fails in fails_vec]
    #violation_count_MB_const = [violation_counts[0][0] for violation_counts in violation_count_vec]

    #var_constants_vec = [i for i in range(3,15)]
    #names = ['MB_evaluate_var_'+str(var) for var in var_constants_vec]#14
    #names_vec = []
    #for name in names:
    #   names_vec.append([[name],[name,None]])
    #reward_vec_indexes,rewards_vec,indexes,fails_vec,var,series_colors,series_names,violation_count_vec = get_data(folder,names_vec,[0],return_violation_count = True)
    #vel_MB_learn = [reward[0][0] for reward in rewards_vec]
    #fail_MB_learn = [fails[0][0] for fails in fails_vec]
    #violation_count_MB_learn = [violation_counts[0][0] for violation_counts in violation_count_vec]

    #names = ['MB_long01']
    #names_vec = []
    #for name in names:
    #   names_vec.append([[name],[name,None]])
    #reward_vec_indexes,rewards_vec,indexes,fails_vec,var,series_colors,series_names,violation_count_vec = get_data(folder,names_vec,[0],return_violation_count = True)
    #vel_MB_long = [reward[0][0] for reward in rewards_vec]
    #fail_MB_long = [fails[0][0] for fails in fails_vec]
    #violation_count_MB_long = [violation_counts[0][0] for violation_counts in violation_count_vec]

    
    plt.figure("Velocity - Fails")
    plt.xlabel('Velocity',fontsize = size)
    plt.ylabel('Fails [%]',fontsize = size)
    #plt.plot(vel_VOD_const,fail_VOD_const,c = 'r',label = 'VOD_const')
    #plt.plot(vel_VOD_const,fail_VOD_const,'o',c = 'r')
    plt.plot(vel_VOD_linear,fail_VOD_linear,c = 'b',label = 'VOD_linear')
    plt.plot(vel_VOD_linear,fail_VOD_linear,'o',c = 'b')
    plt.plot(vel_MB_linear,fail_MB_linear,c = 'g',label = 'MB linear')
    plt.plot(vel_MB_linear,fail_MB_linear,'o',c = 'g')
    #plt.plot(vel_MB_const,fail_MB_const,c = 'orange',label = 'MB const')
    #plt.plot(vel_MB_const,fail_MB_const,'o',c = 'orange')
    #plt.plot(vel_MB_learn,fail_MB_learn,c = 'orange',label = 'MB const')
    #plt.plot(vel_MB_learn,fail_MB_learn,'o',c = 'orange')

    #plt.plot(vel_MB_long,fail_MB_long,'*',c = 'black',label = 'MB_long01')
    #plt.legend()

    plt.figure("Velocity - Violation count")
    plt.xlabel('Velocity',fontsize = size)
    plt.ylabel('Violation count [%]',fontsize = size)
    #plt.plot(vel_VOD_const,violation_count_VOD_const,c = 'r',label = 'VOD_const')
    #plt.plot(vel_VOD_const,violation_count_VOD_const,'o',c = 'r')
    plt.plot(vel_VOD_linear,violation_count_VOD_linear,c = 'b',label = 'VOD_linear')
    plt.plot(vel_VOD_linear,violation_count_VOD_linear,'o',c = 'b')
    #plt.plot(vel_MB_const,violation_count_MB_const,c = 'orange',label = 'VOD_const')
    #plt.plot(vel_MB_const,violation_count_MB_const,'o',c = 'orange')
    plt.plot(vel_MB_linear,violation_count_MB_linear,c = 'g',label = 'MB linear')
    plt.plot(vel_MB_linear,violation_count_MB_linear,'o',c = 'g')
    #plt.plot(vel_MB_long,violation_count_MB_long,'*',c = 'black',label = 'MB_long01')
    #plt.legend()

    plt.figure("Factor - Vel")
    plt.xlabel('safety Factor',fontsize = size)
    plt.ylabel('Vel',fontsize = size)
    ##plt.plot(var_vec_VOD_const,vel_VOD_const,c = 'r',label = 'VOD_const')
    ##plt.plot(var_vec_VOD_const,vel_VOD_const,'o',c = 'r')
    plt.plot(var_vec_VOD_linear,vel_VOD_linear,c = 'b',label = 'VOD_linear')
    plt.plot(var_vec_VOD_linear,vel_VOD_linear,'o',c = 'b')
    plt.plot(var_vec_MB_linear,vel_MB_linear,c = 'g',label = 'MB linear')
    plt.plot(var_vec_MB_linear,vel_MB_linear,'o',c = 'g')
    ##plt.plot(var_vec_MB_long,violation_count_MB_long,'*',c = 'black',label = 'MB_long01')
    #plt.legend()

    plt.figure("Factor - Violation count")
    plt.xlabel('safety Factor',fontsize = size)
    plt.ylabel('Violation count [%]',fontsize = size)
    ##plt.plot(var_vec_VOD_const,violation_count_VOD_const,c = 'r',label = 'VOD_const')
    ##plt.plot(var_vec_VOD_const,violation_count_VOD_const,'o',c = 'r')
    plt.plot(var_vec_VOD_linear,violation_count_VOD_linear,c = 'b',label = 'VOD_linear')
    plt.plot(var_vec_VOD_linear,violation_count_VOD_linear,'o',c = 'b')
    plt.plot(var_vec_MB_linear,violation_count_MB_linear,c = 'g',label = 'MB linear')
    plt.plot(var_vec_MB_linear,violation_count_MB_linear,'o',c = 'g')
    #plt.legend()

    plt.figure("Factor - Fails")
    plt.xlabel('safety Factor',fontsize = size)
    plt.ylabel('Fails [%]',fontsize = size)
    ##plt.plot(var_vec_VOD_const,fail_VOD_const,c = 'r',label = 'VOD_const')
    ##plt.plot(var_vec_VOD_const,fail_VOD_const,'o',c = 'r')
    plt.plot(var_vec_VOD_linear,fail_VOD_linear,c = 'b',label = 'VOD_linear')
    plt.plot(var_vec_VOD_linear,fail_VOD_linear,'o',c = 'b')
    plt.plot(var_vec_MB_linear,fail_MB_linear,c = 'g',label = 'MB linear')
    plt.plot(var_vec_MB_linear,fail_MB_linear,'o',c = 'g')
    #plt.legend()


    #fig, ax1 = plt.subplots()
    #color = 'tab:red'
    #ax1.set_xlabel('Velocity factor',fontsize = size)
    #ax1.set_ylabel('Fails [%]',fontsize = size)
    #ax1.plot(vel,fail,'o',c = color,label = 'VOD')
    #ax1.plot(vel,fail,c = color,)
    #ax1.tick_params(axis='y', labelcolor=color)
    #ax1.plot([1.202],[0.0],'*',c = color,label = 'Model based')

    #ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    #color = 'tab:blue'
    #ax2.set_ylabel('Violation count [%]',fontsize = size, color=color)  # we already handled the x-label with ax1
    #ax2.plot(vel,violation_count, color=color)
    #ax2.plot(vel,violation_count,'o', color=color,label = 'violation_count')
    #ax2.tick_params(axis='y', labelcolor=color)

    #ax2.plot([1.202],[0.06],'*',c = color,label = 'Model based')
    #fig.tight_layout()  # otherwise the right y-label is slightly clipped
    
    plt.show()
if __name__ == "__main__":
    #folder = "new_state"#\old reward backup"
    #folder = "paper_fix"
    size = 15
    names_vec = []

    ##names = ["VOD_00","VOD_002","VOD_004","VOD_006","VOD_008","VOD_01"]
    #folder = "paper_fix"#in REVO paper
    #names = ["VOD_00","VOD_005","VOD_01","VOD_015","VOD_0175","VOD_02"]#in REVO paper
    #for name in names:
    #    names_vec.append([[name],[name,None]])
    #reward_vec_indexes,rewards_vec,indexes,fails_vec,var,series_colors,series_names = get_data(folder,names_vec)
    #vel = [reward[0][0] for reward in rewards_vec]
    #fail = [fails[0][0] for fails in fails_vec]
    
    #vel = [vel[i] / vel[0] for i in range(len(vel))]
    #plt.figure(3)
    #plt.xlabel('Velocity factor',fontsize = size)
    #plt.ylabel('Fails [%]',fontsize = size)
    #plt.plot(vel,fail,'o',c = 'r',label = 'VOD')
    #plt.plot(vel,fail,c = 'r')
    #plt.plot([1.12],[0.6],'o',c = 'green',label = 'REVO')
    #plt.plot([1.11],[0.8],'o',c = 'blue',label = 'REVO+F')
    #plt.plot([1.08],[0.6],'o',c = 'black',label = 'REVO+A')
    #plt.legend()
    #plt.show()
    #names_vec.append(['same_REVO2'])#,'same_REVO2'])#+str(i) for i in range(1,5)])
    #names_vec.append(['eps_REVO1','eps_REVO2','eps_REVO3'])
    #names_vec.append([['batch_256_REVO1','batch_256_REVO2','batch_256_REVO3','batch_256_REVO4','batch_256_REVO5'],['batch_256_REVO','blue']])

    #names_vec.append([['copyREVO+FA3'],['REVO+FA','orange']])


    
    ###################REVO paper final##############################
    #folder = 'REVO_paper_final'
    #names_vec.append([['REVO6','REVO7','REVO8','REVO9','REVO10'],['RL','green']])#corrected 'REVO'
    #names_vec.append([['REVO+A1','REVO+A2','REVO+A3','REVO+A4','REVO+A8'],['RL + Direct','black']])#corrected , 'REVO+A'
    #names_vec.append([['REVO+F1','REVO+F2','REVO+F3','REVO+F4','REVO+F5'],['REVO+F','blue']])#corrected,'REVO+F5' ,'REVO+F2','REVO+F3','REVO+F4' REVO+F1, data_manager_15000 was empty, replaced by 20000
    ####################################################################
    
    #names_vec.append([['REVO+FA1','REVO+FA2' ,'REVO+FA3','REVO+FA4','REVO+FA5'],['REVO+FA','orange']])#corrected   'REVO+FA1' ,'REVO+FA3','REVO+FA4'
    #names_vec.append([['REVO+F1'],['REVO+F1',None]])
    #names_vec.append([['REVO+F2'],['REVO+F2',None]])
    #names_vec.append([['REVO+F3'],['REVO+F3',None]])
    #names_vec.append([['REVO+F4'],['REVO+F4',None]])#important: ,95000? avg_reward: 1.1415095329248326 single_fails: 2.004008016032064 len: 998
    #names_vec.append([['REVO+F5'],['REVO+F5',None]])

    #names_vec.append([['REVO+A1'],['REVO+A1',None]])
    #names_vec.append([['REVO+A2'],['REVO+A2',None]])
    #names_vec.append([['REVO+A3'],['REVO+A3',None]])
    #names_vec.append([['REVO+A4'],['REVO+A4',None]])
    #names_vec.append([['REVO+A8'],['REVO+A8',None]])

    #names_vec.append([['REVO+A4'],['REVO+A4',None]])
    ##names_vec.append([['REVO+A5'],['REVO+A5',None]])
    ##names_vec.append([['REVO+A6'],['REVO+A6',None]])
    ##names_vec.append([['REVO+A7'],['REVO+A7',None]])
    #names_vec.append([['REVO+A8'],['REVO+A8',None]])

    #names_vec.append([['REVO8'],['REVO','green']])

    #names_vec.append([['REVO10'],['REVO','green']])# REVO10   85000 avg_reward: 1.1851330422794382 single_fails: 0.5 len: 1000

    #names_vec.append([['REVO+FA6','REVO+FA7','REVO+FA8','REVO+FA9','REVO+FA10'],['REVO+FA','blue']])
    #names_vec.append([["same_REVO3"],['REVO','green']])#"same_REVO1" "same_REVO2",,"same_REVO5" ,"same_REVO4"
    #names_vec.append([["same_REVO+F5",],['same_REVO+F1','blue']])


    ################################final#########################
    
    #names_vec.append([['REVO10'],['REVO','green']])#85000
    #names_vec.append([['REVO+A3'],['REVO3',None]])#90000
    #names_vec.append([['REVO+F3'],['REVO+F3',None]])#70000
    #names_vec.append([['VOD_long2'],['VOD',None]])#0
    
    #names_vec.append([['also_steer1'],['REVO',None]])#90000
    ###########################model based 25.9.19############
    #plot_fails_vel_comparison()

    ########################################################################
    folder = "MB_paper_stabilize"
    #plot_fails_vel_comparison()
    baseline_folder = "MB_paper_stabilize"
    baseline = 'VOD_baseline_0.07'#"baseline"
    names_vec = []
    #names = ['MB_var_check_linear2_'+str(var_constant) for var_constant in [0.01*i for i in range(0,15)]]#14
    #for name in names:
    #   names_vec.append([[name],[name,None]])
    #names = ['MB_var_check_const2_'+str(var_constant) for var_constant in [0.01*i for i in range(0,20)]]#14
    #for name in names:
    #   names_vec.append([[name],[name,None]])
    #names = ['VOD_var_check_const2_'+str(var_constant) for var_constant in [0.01*i for i in range(0,20)]]
    #for name in names:
    #   names_vec.append([[name],[name,None]])
    #names = ['VOD_var_check_linear2_'+str(var_constant) for var_constant in [0.01*i for i in range(0,15)]]
    #for name in names:
    #   names_vec.append([[name],[name,None]])
    #names = ['MB_evaluate_var_'+str(var_constant) for var_constant in [i for i in range(3,15)]]
    #for name in names:
    #   names_vec.append([[name],[name,None]])
    #comp_relative_reward(folder,names_vec,[0],baseline)#[5000*j for j in range(1,21)])

    #add_zero_data_manager(folder,names_vec)
    
    #plot_fails_vel_comparison()

    #names_vec = []
    #names_vec.append([["MB_learn_var_3_actions_2"],['MB',None]])#'REVO_direct_reward_3','REVO_direct_rew ard_4' MB_evaluate_var_10
    #train_indexes = [0]
    ##train_indexes = [100*j for j in range(1,20)]
    #reward_vec_indexes,rewards_vec,indexes,fails_vec,var,series_colors,series_names, _ = get_data(folder,names_vec,train_indexes,return_violation_count = True)
    ##indexes = [ind/2 for ind in indexes]#
    #avg_rewards_vec,var_rewards_vec = average_training_processes(rewards_vec)
    #avg_fails_vec,var_fails_vec = average_training_processes(fails_vec)
    
    folder = 'MB_paper_stabilize'#/MB_learning_process2'
    names_vec = []
    #train_indexes = list(range(50))
    #names = ['VOD_learning_process_'+str(num) for num in train_indexes]
    #names = ['MB_learning_process_no_stabilize1_'+str(num) for num in train_indexes]
    #train_indexes = list(range(50))
    #names = ['MB_learning_process2/MB_learning_process_no_stabilize1_'+str(num) for num in train_indexes]
    #names_vec.append([names,['MB','green']])

    #train_indexes = list(range(50))
    #names = ['MB_learning_process1MB_learning_process_'+str(num) for num in train_indexes]
    #names_vec.append([names,['MB stabilize','green']])#'REVO_direct_reward_3','REVO_direct_rew ard_4' MB_evaluate_var_10


    #train_indexes = list(range(50))
    #names = ['MB_learning_process1_0.05/MB_learning_process_stabilize1_0.05_'+str(num) for num in train_indexes]
    #names_vec.append([names,['stabilize 0.05','blue']])
    train_indexes = list(range(6))
    names = ['MB_learning_process_no_stabilize1_0.05/MB_learning_process_no_stabilize1_0.05_'+str(num) for num in train_indexes]
    names_vec.append([names,['stabilize','blue']])

    train_indexes = list(range(50))
    names = ['MB_learning_process0.05/MB_learning_process_stabilize_0.05_'+str(num) for num in train_indexes]
    names_vec.append([names,['stabilize','black']])





    reward_vec_indexes,avg_rewards_vec,indexes,avg_fails_vec,var_rewards_vec,series_colors,series_names, violation_count_vec,var_fails_vec,stabilize_count_brake_vec,stabilize_count_steer_vec = get_training_process_data(folder,names_vec,train_indexes,baseline_folder,baseline)
    #indexes = [ind/2 for ind in indexes]#
    #avg_rewards_vec,var_rewards_vec = average_training_processes(rewards_vec)
    #avg_fails_vec,var_fails_vec = average_training_processes(fails_vec)

    
   

    xlabel = 'Episodes' #'Time steps'
    plt.figure(1)
    plt.tick_params(labelsize=12)
    plt.ylim(0,1.5)
    plt.xticks(indexes)
    plt.xlabel(xlabel,fontsize = size)#'Train iterations number'
    plt.ylabel('Normalized average velocity' ,fontsize = size)#'Relative progress'
    #for rewards,color in zip(rewards_vec,series_colors):
    #    for single_rewards in rewards:
    #        plt.scatter(trains[:len(single_rewards)],single_rewards,c = color,alpha = 0.5)#
    #plt.xlim(0,100000)
    for reward_indexes,rewards,var,color,label in zip(reward_vec_indexes,avg_rewards_vec,var_rewards_vec,series_colors,series_names):
        plt.plot(indexes[:len(rewards)],rewards,color = color,label = label)
        plt.errorbar(indexes[:len(rewards)],rewards,var,c = color,alpha = 0.7)#reward_indexes[0]
        #print("var:",var[0])
    plt.plot([0,indexes[len(rewards)-1]],[1,1],linewidth = 2.0,c = 'r',label = 'Direct')#'VOD'
    plt.legend()

    plt.figure(2)
    plt.tick_params(labelsize=12)
    plt.xticks(indexes)
    #plt.xlim(0,100000)
   # plt.ylim(0,100)
    plt.xlabel(xlabel,fontsize = size)
    plt.ylabel('Fails [%]',fontsize = size)
    #for fails,color in zip(fails_vec,series_colors):
    #    for single_fails in fails:
    #        plt.scatter(indexes[:len(single_fails)],single_fails,c = color,alpha = 0.5)
    for fails,var,color,label in zip(avg_fails_vec,var_fails_vec,series_colors,series_names):
        plt.plot(indexes[:len(fails)],fails,color = color,label = label)
        #plt.errorbar(indexes[:len(fails)],fails,var,c = color,alpha = 0.7)
    plt.legend()

    plt.figure(3)
    plt.tick_params(labelsize=12)
    plt.xticks(indexes)
    #plt.xlim(0,100000)
    #plt.ylim(0,100)
    plt.xlabel(xlabel,fontsize = size)
    plt.ylabel('Violation count [%]',fontsize = size)
    #for fails,color in zip(fails_vec,series_colors):
    #    for single_fails in fails:
    #        plt.scatter(indexes[:len(single_fails)],single_fails,c = color,alpha = 0.5)
    for violation_count,var,color,label in zip(violation_count_vec,var_fails_vec,series_colors,series_names):
        plt.plot(indexes[:len(violation_count)],violation_count,color = color,label = label)
        #plt.errorbar(indexes[:len(fails)],fails,var,c = color,alpha = 0.7)
    plt.legend()

    plt.figure(4)
    plt.tick_params(labelsize=12)
    plt.xticks(indexes)
    #plt.xlim(0,100000)
    #plt.ylim(0,100)
    plt.xlabel(xlabel,fontsize = size)
    plt.ylabel('Stabilization [%]',fontsize = size)
    #for fails,color in zip(fails_vec,series_colors):
    #    for single_fails in fails:
    #        plt.scatter(indexes[:len(single_fails)],single_fails,c = color,alpha = 0.5)
    #for stabilize_count_brake,stabilize_count_steer,color,label in zip(stabilize_count_brake_vec,stabilize_count_steer_vec,series_colors,series_names):
    for stabilize_count_brake,stabilize_count_steer,color,label in zip(stabilize_count_brake_vec[1:],stabilize_count_steer_vec[1:],series_colors[1:],series_names[1:]):
        plt.plot(indexes[:len(stabilize_count_brake)],stabilize_count_brake,color = None,label = label+' braking')
        plt.plot(indexes[:len(stabilize_count_steer)],stabilize_count_steer,color = None,label = label+ ' steering')
        #plt.errorbar(indexes[:len(fails)],fails,var,c = color,alpha = 0.7)
    plt.legend()

    




    plt.show()
