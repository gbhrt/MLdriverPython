import matplotlib.pyplot as plt
import numpy as np
import data_manager1
from hyper_parameters import HyperParameters
import os
import library as lib
import environment1
import classes

def show_path(name):
    HP = HyperParameters()
    envData = environment1.OptimalVelocityPlannerData(env_mode = "DDPG")
    HP.restore_name = name
    HP.save_name = name
    save_path = os.getcwd()+ "\\files\\models\\no_friction_hard\\"+HP.save_name+"\\"
    restore_path = os.getcwd()+ "\\files\\models\\no_friction_hard\\"+HP.restore_name+"\\"
    dataManager = (data_manager1.DataManager(save_path,restore_path,True))

    #if len(self.real_path.time) > 0:
    #    for i in range(1,len(self.real_path.time)):
    #        self.real_path.time[i] -= self.real_path.time[0]
    #    self.real_path.time[0] = 0.0
    #    path = compute_analytic_path(self.path_seed[-1])

    plt.figure(1)
    size = 15
    #analytic_path = lib.compute_analytic_path(1111)
    #max_dis_ind = 0
    #for j,dis in enumerate (analytic_path.distance):
    #    max_dis_ind = j
    #    if dis > 300:
    #        break


    

    for i in range(0,len(dataManager.paths),1):
        if dataManager.path_seed[i] != 1111 or i*5+5 != 855:
            continue
        analytic_path = lib.compute_analytic_path(dataManager.path_seed[i])# 1111 #dataManager.path_seed[i])
        max_dis_ind = 0
        for j,dis in enumerate (analytic_path.distance):
            max_dis_ind = j
            #if dis > 300:
            if dis > dataManager.paths[i][1][-1]:
                break
        #plt.title("episode number: "+str(i*5+5),fontsize = size)
        print("run:",i*5+5)
        plt.plot(dataManager.paths[i][1],dataManager.paths[i][0],c = "orange", label = "REVO+FA")#'tab:purple' (1.0,1.0,0.0)
        #plt.plot(dataManager.paths[i][1],dataManager.paths[i][0])
        #plt.plot(np.array(analytic_path.distance)[:max_dis_ind],np.array(analytic_path.analytic_velocity_limit)[:max_dis_ind],c = (0.0,0.0,0.0))#linewidth = 1.0
        #plt.plot(np.array(analytic_path.distance)[:max_dis_ind],np.array(analytic_path.analytic_velocity)[:max_dis_ind],c = 'r')#,linewidth = 1.0
        plt.xlabel('distance [m]',fontsize = size)
        plt.ylabel('velocity [m/s]',fontsize = size)

        plt.plot(np.array(analytic_path.distance)[:max_dis_ind],np.array(analytic_path.analytic_velocity)[:max_dis_ind],c = 'r',label = "VOD")#,linewidth = 1.0
        #plt.figure(2)
        #plt.plot(np.array(analytic_path.position)[:,0][:max_dis_ind],np.array(analytic_path.position)[:,1][:max_dis_ind], 'g')
        #plt.xlabel('x [m]',fontsize = size)
        #plt.ylabel('y [m]',fontsize = size)
        plt.legend()
        plt.show()

if __name__ == "__main__": 
    name = "bevo_1"#"add_analytic_feature_random_2"#"final_1"  "final_random_1" - final_analytic_random_1 not clear
    show_path(name)

