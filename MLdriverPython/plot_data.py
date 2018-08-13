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

def plot_rewards(names,shape=None,color=None):
    max_train_num = 1000
    episodes_num = 50
    dataManager_vec = []
    relative_rewards_changed_vec = []
    for i in range(len(names)):
        HP.restore_name = names[i]
        HP.save_name = names[i]
        save_path = os.getcwd()+ "\\files\\models\\final3\\"+HP.save_name+"\\"
        restore_path = os.getcwd()+ "\\files\\models\\final3\\"+HP.restore_name+"\\"
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
    combined_rewards = []
    for i in range(len(dataManager_vec)):
        for j in range(len(relative_rewards_changed_vec[i])):
            if relative_rewards_changed_vec[i][j] != None:
                combined_rewards.append([dataManager_vec[i].train_num[j],relative_rewards_changed_vec[i][j]])
    combined_rewards = np.array(sorted(combined_rewards, key=lambda x: x[0]))

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
    N = 20
    for i in range(len(fails_vec)):
        fails_density = lib.running_average(fails_vec[i],N)
        fails_density_vec.append(fails_density)

    combined_fails_density = []
    for i in range(len(dataManager_vec)):
        for j in range(len(fails_density_vec[i])):
            combined_fails_density.append([dataManager_vec[i].train_num[j],fails_density_vec[i][j]])
    combined_fails_density = np.array(sorted(combined_fails_density, key=lambda x: x[0]))


    combined_abs_rewards = []
    for i in range(len(dataManager_vec)):
        for j in range(len(relative_rewards_changed_vec[i])):
            #combined_abs_rewards.append([dataManager_vec[i].train_num[j],sum(dataManager_vec[i].paths[j][0])*0.2])
            combined_abs_rewards.append([dataManager_vec[i].train_num[j],dataManager_vec[i].paths[j][1][-1]])
    combined_abs_rewards = np.array(sorted(combined_abs_rewards, key=lambda x: x[0]))

    plt.figure(1)
    for j in range(len(relative_rewards_changed_vec)):
        plt.scatter(np.array(dataManager_vec[j].train_num)[:max_train_num],(relative_rewards_changed_vec[j])[:max_train_num],marker = shape,alpha = 0.2)#,c = color
    ave = lib.running_average(combined_rewards[:,1],20)[:max_train_num]
    plt.plot((combined_rewards[:,0])[:len(ave)],ave,c = color)

    ones = np.ones([len(dataManager_vec[max_len_ind].train_num)])[:max_train_num]
    plt.plot(dataManager_vec[max_len_ind].train_num[:max_train_num],ones,linewidth = 2.0,c = 'r')

    plt.figure(2)
    for j in range(len(relative_rewards_changed_vec)):
        abs_rewards = [dataManager_vec[j].paths[k][1][-1] for k in range(len(dataManager_vec[j].paths))]
        abs_rewards1 = [sum(dataManager_vec[j].paths[k][0])*0.2 for k in range(len(dataManager_vec[j].paths))]#for debugging
        #for w in range(len(abs_rewards)):
            #print("sum vel:",abs_rewards1[w],"dis:",abs_rewards[w])
            #print(dataManager_vec[j].paths[w][1])

        abs_rewards = abs_rewards[:max_train_num]
        plt.scatter(np.array(dataManager_vec[j].train_num)[:max_train_num],abs_rewards,marker = shape,alpha = 0.2)#c = color
    ave = lib.running_average(combined_abs_rewards[:,1],20)[:max_train_num]
    plt.plot((combined_abs_rewards[:,0])[:len(ave)],ave,c = color)

    analytic_dist_vec = [analytic_path.distance[max_tim_ind] for _ in range(len(dataManager_vec[max_len_ind].train_num))]
    analytic_dist_vec = analytic_dist_vec[:max_train_num]
    plt.plot(dataManager_vec[max_len_ind].train_num[:max_train_num],analytic_dist_vec,linewidth = 2.0,c = 'r')

    plt.figure(3)
    #for i in range(len(fails_density_vec)):
    #    plt.plot(np.array(dataManager_vec[i].train_num)[:len(fails_density_vec[i])],(fails_density_vec[i]))
    ave = lib.running_average(combined_fails_density[:,1],20)[:max_train_num]
    plt.plot((combined_fails_density[:,0])[:len(ave)],ave*100,c = color)

if __name__ == "__main__":
    #convolution:
    #names1 = ["final_conv_analytic_new_reward_2","final_conv_analytic_new_reward_4","final_conv_analytic_new_reward_6","final_conv_analytic_new_reward_8"]#,"
    #names2 = ["final_conv_new_reward_1","final_conv_new_reward_2","final_conv_new_reward_3","final_conv_new_reward_4"]
    #names1 = ["final_conv_new_reward_same_1"]
    #names2 = ["final_conv_analytic_new_reward_same_1"]#,"final_conv_analytic_new_reward_same_3","final_conv_analytic_new_reward_same_5"]
    ############################################################

    #same path for testing and training. 2 trains per step time:
    #analytic feature:
    #names1 = ["compare_same_run_path//final_analytic_2","compare_same_run_path//final_analytic_6",\
    #    "compare_same_run_path//final_analytic_8","compare_same_run_path//final_analytic_10"]#seed = 1111
    ##without analytic feature:
    #names2 = ["compare_same_run_path//final_1","compare_same_run_path//final_3","compare_same_run_path//final_5",\
    #    "compare_same_run_path//final_7","compare_same_run_path//final_9"]#seed = 1111
    ##################################################

    #random path for testing. random for training. 2 trains per step time:  - a very little different between names1 and names2 
    #analytic feature:
    #names1 = ["final_2_random","final_4_random","final_6_random","final_8_random","final_10_random"]
    #without analytic feature:
    #names2 = ["final_analytic_2_random","final_analytic_4_random","final_analytic_6_random","final_analytic_8_random","final_analytic_10_random"]
    #########################################################################################

    #same path for testing. random for training. 2 trains per step time:  - a very little different between names1 and names2 
    #analytic feature:
    #names1 = ["final_analytic_random_1","final_analytic_random_3"] #5 short 7 empty
    #without analytic feature:
    #names2 = ["final_random_1","final_random_3","final_random_5"] 
    #names1 = ["final_random_2","final_random_4","final_random_6"] 
    #########################################################################################

    
    #add analytic action to action. random paths, test same path (1111) 2 trains per step time:
    #analytic feature:
    #names1 = ["add_analytic_feature_random_2","add_analytic_feature_random_4","add_analytic_feature_random_6","add_analytic_feature_random_8","add_analytic_feature_random_0"]
    ## 900 630 592 496
    ## no analytic feature:
    #names2 = ["add_analytic_random_1","add_analytic_random_2","add_analytic_random_3","add_analytic_random_4","add_analytic_random_5"]\
        #,"add_analytic_random_5","add_analytic_random_7","add_analytic_random_6","add_analytic_random_8"]#not a big difference

   # names1 = ["add_analytic_random_5"]#,"add_analytic_random_7","add_analytic_random_6","add_analytic_random_8"]
    #417 496 417 496
    ###########################################################################################



    #names1 = ["final_random_1","final_random_3"]
    #names2 = ["final_analytic_random_1","final_analytic_random_3"]
    #final_3 - short
    #names2 = ["final_analytic1111_2","final_analytic1111_6","final_analytic1111_8","final_analytic1111_10"]
    #names2 = ["final_2"]
    #names3 = ["final_analytic_1","final_analytic_3","final_analytic_5","final_analytic_7"]#seed  = 1236


    #same path (seed = 1111) for testing. different training steps per step time:
    #names1 = ["final_analytic_1_4","final_analytic_3_4"]
    #names2 = ["final_11_4","final_13_4_short","final_15_4","final_17_4","final_19_4"]

    #random path for testing: different training steps per step time:
    #names1 = ["final_analytic_2_10","final_analytic_4_10","final_analytic_6_10","final_analytic_8_10","final_analytic_10_10"]#,
    #names2 = ["final_4_10","final_6_10"]#,"final_1
    

    #names1 = ["final1_analytic_action_1_0","final_analytic_action_1_1","final_analytic_action_1_2"]


    ############################################################################################################
    # final1:
    #"regular5" "regular7" to short
    #,"add_acc_feature4" "add_acc_feature8", to short

    #names1 = ["regular9","regular11","regular13"]#,"regular3"  ["regular1"]#,
    #names2 = ["regular3"]
    #names3 = ["regular11"]
    #names4 = ["regular13"]
    #names5 = ["regular9"]
    #names1 = ["regular2","regular4","regular6","regular8"]
    #names1 = ["regular_uni2","regular_uni4","regular_uni6"]
    #names2 = ["add_acc_feature2_uni","add_acc_feature4_uni","add_acc_feature6_uni"]
    

   # names2 =["add_acc_feature2","add_acc_feature6","add_acc_feature10","add_acc_feature12","add_acc_feature14"]
    #names3 =["add_acc2","add_acc6","add_acc8","add_acc10","add_acc12","add_acc14"]

    ################################################33
    names1 = ["regular1","regular3","regular5","regular7","regular9"]
   

    names1 = ["regular5_1"]
    plot_rewards(names1,'o',(0.0,0.0,0.0))#'b' 'tab:purple'
    #plot_rewards(names2)
    #plot_rewards(names3)
    #plot_rewards(names4)
    #plot_rewards(names5)
    #plot_rewards(names2,'x','g')
    #plot_rewards(names3,'.','b')
    size = 15
    plt.figure(2)
    plt.xlabel('Train iterations number',fontsize = size)
    plt.ylabel('Progress on the path [m]',fontsize = size)
    plt.figure(1)
    plt.xlabel('Train iterations number',fontsize = size)
    plt.ylabel('Relative mean velocity',fontsize = size)
    plt.figure(3)
    plt.xlabel('Train iterations number',fontsize = size)
    plt.ylabel('Fails [%]',fontsize = size)
    plt.show()
