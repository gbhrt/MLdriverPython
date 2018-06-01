import plot
import classes
import numpy as np
import json
import plot_lib 
import matplotlib.pyplot as plt
import numpy as np
import pathlib
import library as lib
import sys
class DataManager():
    def __init__(self,save_path = None, restore_path = None,restore_flag = False):
        if save_path != None:
            save_path += "data_manager\\"
            pathlib.Path(save_path).mkdir(parents=True, exist_ok=True) 
            self.save_name = save_path+"data.txt"
            self.save_readable_data_name = save_path+"readable_data.txt"
        else:
            self.save_name ="data.txt"
        if restore_path != None:
            self.restore_name = restore_path+"data_manager\\data.txt"
        else:
            self.restore_name = "data.txt"
        
        #define data variables 
        self.real_path = classes.Path()
        self.planned_path = classes.Path()
        self.rewards = []
        self.lenght = []
        self.relative_reward = []
        self.episode_end_mode = []
        self.path_seed = []
        self.run_num = []
        self.init_run_num = 0
        self.train_num = []
        self.init_train_num = 0
        
        

        #reset every episode:
        self.Qa = []
        self.Q0 = []
        self.Q1 = []
        self.Qneg1 = []
        self.roll = []
        self.noise = []
        self.acc = []
        ###############################
        
        if restore_flag:
            self.load_data()
        if len(self.run_num) > 0:
            self.init_run_num = self.run_num[-1]
        if len(self.train_num) > 0:
            self.init_train_num = self.train_num[-1]
    def add_run_num(self,i):
        self.run_num.append(i+self.init_run_num)
    def add_train_num(self,i):
        self.train_num.append(i+self.init_train_num)

    def plot_all(self):
       # try:
            plt.figure(1,figsize = (10,5))
            plt.subplot(221) 
            #print("plot all")
            plot_lib.plot_path(self.real_path)
            #plt.plot(self.real_path.distance,np.array(self.acc)*10)
            #plt.plot(self.real_path.distance,np.array(self.noise)*10)
            plt.subplot(222)  
            plt.plot(self.roll)
            #x = np.array(self.real_path.distance)
            #Qa, = plt.plot(x,self.Qa,label = "Qa")
            #Q0, = plt.plot(x,self.Q0,label = "Q0")
            #Q1, = plt.plot(x,self.Q1,label = "Q1")
            #Qneg1, = plt.plot(x,self.Qneg1,label = "Qneg1")
            #plt.legend(handles=[Qa, Q0,Q1,Qneg1])

            plt.subplot(223)  
            
            plt.title("train num - relative reward")
            col = []
            for mode in self.episode_end_mode:
                if mode == 'kipp' or mode == 'deviate':
                    col.append('r')
                else:
                    col.append('g')
            relative_reward_zero = list(self.relative_reward)
            for i in range(len(self.episode_end_mode)):
                if self.episode_end_mode[i] == 'kipp' or self.episode_end_mode[i] == 'deviate':
                    relative_reward_zero[i] = -2.0
            plt.scatter(self.train_num,relative_reward_zero,c = col)
            #plt.title("episodes reward")
            #plt.plot(self.run_num,self.rewards,'o')
            #if len(self.run_num) >= 15:
            #    ave = lib.running_average(self.rewards,15)
            #    plt.plot(self.run_num[:len(ave)],ave)
            #plt.plot(self.run_num,lib.running_average(self.rewards,100))
        
            plt.subplot(224)  
            plt.title("episodes - relative reward")
            col = []
            for mode in self.episode_end_mode:
                if mode == 'kipp' or mode == 'deviate':
                    col.append('r')
                else:
                    col.append('g')
            relative_reward_zero = list(self.relative_reward)
            for i in range(len(self.episode_end_mode)):
                if self.episode_end_mode[i] == 'kipp' or self.episode_end_mode[i] == 'deviate':
                    relative_reward_zero[i] = -2.0
            plt.scatter(self.run_num,relative_reward_zero,c = col)
            #if len(self.run_num) >= 50:
            #    ave = lib.running_average(relative_reward_zero,50)
            #    plt.plot(self.run_num[:len(ave)],ave)
            #plt.plot(lib.running_average(self.relative_reward,5))

            #plt.title("episodes lenght")
            #plt.plot(self.lenght)

            #plt.figure(2)
            #plt.plot(np.array(self.planned_path.position)[:,0],np.array(self.planned_path.position)[:,1])
            #if len(self.real_path.position) > 0:
            #    plt.plot(np.array(self.real_path.position)[:,0],np.array(self.real_path.position)[:,1])

            plt.show()
        #except:
        #    print ("error in plot data:", sys.exc_info()[0])
        #    raise
            return

    def update_planned_path(self,path):
        self.planned_path = path
        return

    def update_real_path(self,pl, velocity_limit = None, analytic_vel = None,curvature = None):
        #if len(self.real_path.position) > 0:
        #    dist = lib.dist(pl.simulator.vehicle.position[0],pl.simulator.vehicle.position[1],self.real_path.position[-1][0],self.real_path.position[-1][1])
        #else:
        #    dist = 0.0
        dist = pl.in_vehicle_reference_path.distance[pl.main_index]
        self.real_path.distance.append(dist)
        self.real_path.position.append(pl.simulator.vehicle.position)
        self.real_path.velocity.append(pl.simulator.vehicle.velocity)
        self.real_path.time.append(pl.get_time())
        if velocity_limit != None: self.real_path.analytic_velocity_limit.append(velocity_limit)
        if analytic_vel != None: self.real_path.analytic_velocity.append(analytic_vel)
        if curvature != None: self.real_path.curvature.append(curvature*50.0)
        return

    def restart(self):
        self.real_path = classes.Path()
        self.Qa = []
        self.Q0 = []
        self.Q1 = []
        self.Qneg1 = []
        self.roll = []
        self.noise = []
        self.acc = []
        
    def save_readeable_data(self):
      #  try: 
            with open(self.save_readable_data_name, 'w') as f:
                for i in range(len(self.rewards)):
                    f.write("%s \t %s\t %s \t %s \n" % (i,self.rewards[i],self.relative_reward[i],self.episode_end_mode[i]))
        #except:
        #    print("cannot save data")
    def save_data(self):
        try: 
            with open(self.save_name, 'w') as f:
                json.dump((self.run_num,self.rewards,self.lenght,self.relative_reward, self.episode_end_mode,self.path_seed ),f)
            print("data manager saved")            
        except:
            print("cannot save data manager")

    def load_data(self):
        try:
            with open(self.restore_name, 'r') as f:
                self.run_num,self.rewards,self.lenght,self.relative_reward, self.episode_end_mode,self.path_seed  = json.load(f)#
            print("data manager restored")
        except:
            print("cannot restore data manager")
