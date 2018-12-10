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
            self.save_run_data_name = save_path+"run_data.txt"
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
        self.paths = []
        self.run_data =[]
        
        
        

        #reset every episode:
        self.restart()
        #self.Qa = []
        #self.Q0 = []
        #self.Q1 = []
        #self.Qneg1 = []
        #self.Qa_target = [] 
        #self.Q0_target = [] 
        #self.Q1_target = [] 
        #self.Qneg1_target = []
        #self.roll = []
        #self.noise = []
        #self.acc = []
        #self.acc_target = []
        #self.wheels_vel = []
        #self.time_stamps = []
        ###############################
        
        if restore_flag:
            self.load_data()
        if len(self.run_num) > 0:
            self.init_run_num = self.run_num[-1]
        if len(self.train_num) > 0:
            self.init_train_num = self.train_num[-1]

    def restart(self):
        self.real_path = classes.Path()
        self.Qa = []
        self.Q0 = []
        self.Q1 = []
        self.Qneg1 = []
        self.Qa_target = [] 
        self.Q0_target = [] 
        self.Q1_target = [] 
        self.Qneg1_target = []
        self.roll = []
        self.noise = []
        self.acc = []
        self.acc_target = []
        self.wheels_vel = []
        self.time_stamps = []
        self.input_time = []
        self.step_times = []
        print("data manager restarted")
    def add_run_num(self,i):
        self.run_num.append(i+self.init_run_num)
    def add_train_num(self,i):
        self.train_num.append(i+self.init_train_num)

    def plot_all(self):
        try:
            #comp analytic path:
            ##if len(self.real_path.time) > 0:
            ##    for i in range(1,len(self.real_path.time)):
            ##        self.real_path.time[i] -= self.real_path.time[0]
            ##    self.real_path.time[0] = 0.0
            ##    path = lib.compute_analytic_path(self.real_path.seed)
            ##    max_time_ind = 0
            ##    for i,tim in enumerate (path.analytic_time):
            ##        max_time_ind = i
            ##        if tim > self.real_path.time[-1]:
            ##            break
            max_time_ind = len(self.real_path.time)
            path = self.real_path
            ###########################

            plt.figure(1,figsize = (10,5))
            plt.subplot(221) 
            #print("plot all")
            #self.plot_path(self.real_path)
            #plt.plot(self.real_path.distance,np.array(self.acc)*10)
            #plt.plot(self.real_path.distance,np.array(self.noise)*10)
            if len(self.real_path.time) > 0:
                real_vel, = plt.plot(np.array(path.distance)[:max_time_ind],np.array(path.analytic_velocity_limit)[:max_time_ind],label = "velocity limit")
                vel_lim,  = plt.plot(np.array(path.distance)[:max_time_ind],np.array(path.analytic_velocity)[:max_time_ind],label = "analytic velocity")
                vel, =      plt.plot(self.real_path.distance,self.real_path.velocity,'o', label = "vehicle velocity")
                plt.legend(handles=[real_vel, vel_lim,vel])

            plt.subplot(222) 
            #print("acc_vec:",self.acc)
            #plt.plot(self.real_path.distance[:max_time_ind-1],self.acc[1:]) #action is before the saved distance
            #plt.plot(self.real_path.distance[:max_time_ind-1],self.acc[1:],'o')
           # plt.plot(self.real_path.distance[:max_time_ind],self.acc_target)
            #print("self.real_path.time[-1] in plot",self.real_path.time[-1])
            #print("analytic_dist in plot:",path.distance[max_time_ind])
            ##if len(self.real_path.time) > 0:
            ##    plt.plot(np.array(path.analytic_time)[:max_time_ind],np.array(path.analytic_velocity_limit)[:max_time_ind])
            ##    plt.plot(np.array(path.analytic_time)[:max_time_ind],np.array(path.analytic_velocity)[:max_time_ind])
          

            ##    plt.plot(self.real_path.time,self.real_path.velocity)
            #plot roll:
            plt.plot(path.distance[:max_time_ind],self.roll[:max_time_ind])
            


            plt.subplot(223)  
            dtime_stamps,dinput_time,dstep_times = [],[],[]
            for i in range(1,len(self.time_stamps)):
                dtime_stamps.append(self.time_stamps[i] - self.time_stamps[i-1])
                dinput_time.append(self.input_time[i] - self.input_time[i-1])
                dstep_times.append(self.step_times[i] - self.step_times[i-1])
            plt.plot(dtime_stamps,label = "stamps")
            plt.plot(dinput_time,label ="input")
            plt.plot(dstep_times,label = "steps")
            plt.legend()

            #if len(self.real_path.time) > 0:
             

            #    x = np.array(self.real_path.distance[:max_time_ind])
            #    Qa, = plt.plot(x,self.Qa,label = "Qa")
            #    Q0, = plt.plot(x,self.Q0,label = "Q0")
            #    Q1, = plt.plot(x,self.Q1,label = "Q1")
            #    Qneg1, = plt.plot(x,self.Qneg1,label = "Qneg1")
            #    plt.legend(handles=[Qa, Q0,Q1,Qneg1])

                #plt.plot(x,np.array(self.wheels_vel)[:,0])
                #plt.plot(x,np.array(self.wheels_vel)[:,1])
                #plt.plot(x,np.array(self.wheels_vel)[:,2])
                #plt.plot(x,np.array(self.wheels_vel)[:,3])
                

            #plt.title("train num - relative reward")
            
            #col = []
            #for mode in self.episode_end_mode:
            #    if mode == 'kipp' or mode == 'deviate':
            #        col.append('r')
            #    else:
            #        col.append('g')
            #relative_reward_zero = list(self.relative_reward)
            #relative_reward_success = []
            #relative_reward_success_ind = []
            #fails = 0
            #fails_range = 50
            #fails_num = []
            #fail_num_ind = []
            #for i in range(len(self.episode_end_mode)):
            #    if self.episode_end_mode[i] == 'kipp' or self.episode_end_mode[i] == 'deviate':
            #        relative_reward_zero[i] = 0.0
            #        fails+=1
            #    else:
            #        relative_reward_success.append(self.relative_reward[i])
            #        relative_reward_success_ind.append(self.train_num[i])
            #    if i % fails_range == 0 and i != 0:
            #        fails_num.append(fails/fails_range)
            #        fail_num_ind.append(self.train_num[i])
            #        fails = 0
            
            ##plt.scatter(relative_reward_success_ind[:len(fails_num)],fails_num)
            ##plt.scatter(self.train_num,relative_reward_zero,c = col)
            #plt.scatter(relative_reward_success_ind,relative_reward_success,c = 'r')
            ##ave = lib.running_average(relative_reward_success,50)
            ##plt.plot(relative_reward_success_ind[:len(ave)],ave,c = 'b')


            #plt.scatter(fail_num_ind,fails_num,c = 'g')
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
                    relative_reward_zero[i] = -0.0
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
        except:
            print ("error in plot data:", sys.exc_info()[0])
            raise
            return
    def plot_path(self,path):
    

        datax = np.array(path.distance)
        #datax = np.array(path.time)
        datay1 = np.array(path.velocity)
        datay2 = np.array(path.analytic_velocity_limit)
        datay3 = np.array(path.analytic_velocity)
        datay4 = np.array(path.curvature)

        real_vel, = plt.plot(datax, datay1,label = "vehicle velocity")
        vel_lim, = plt.plot(datax, datay2,label = "velocity limit")
        vel, =   plt.plot(datax, datay3, label = "analytic velocity")
        #curv, = plt.plot(datax, datay4, label = "curvature")
        plt.legend(handles=[real_vel, vel_lim,vel])#,curv
        #plt.show()

        return

    def update_planned_path(self,path):
        self.planned_path = path
        return

    def update_real_path(self,pl, velocity_limit = None, analytic_vel = None,curvature = None,seed = None):
        #if len(self.real_path.position) > 0:
        #    dist = lib.dist(pl.simulator.vehicle.position[0],pl.simulator.vehicle.position[1],self.real_path.position[-1][0],self.real_path.position[-1][1])
        #else:
        #    dist = 0.0
        dist = pl.in_vehicle_reference_path.distance[pl.main_index]
        self.real_path.distance.append(dist)
        self.real_path.position.append(pl.simulator.vehicle.position)
        self.real_path.velocity.append(pl.simulator.vehicle.velocity[1])
        self.real_path.time.append(pl.get_time())
        if velocity_limit != None: self.real_path.analytic_velocity_limit.append(velocity_limit)
        if analytic_vel != None: self.real_path.analytic_velocity.append(analytic_vel)
        if curvature != None: self.real_path.curvature.append(curvature*50.0)
        if seed != None: self.real_path.seed = seed
        return
    def comp_relative_reward(self):
        print("relative reward seed:",self.real_path.seed)
        if self.episode_end_mode[-1] == 'kipp' or self.episode_end_mode[-1] == 'deviate' or len(self.real_path.distance) == 0:
            return None
        #dist = sum(self.real_path.velocity)*0.2
        dist = self.real_path.distance[-1]
        print("dist",dist)
        path = classes.Path()
        path.position = lib.create_random_path(9000,0.05,seed = self.real_path.seed)
        
        lib.comp_velocity_limit_and_velocity(path,skip = 10,reduce_factor = 1.0)
        max_time_ind = 0
        for i,tim in enumerate (path.analytic_time):
            max_time_ind = i
            if tim > self.real_path.time[-1] - self.real_path.time[0]:
                break
        analytic_dist = path.distance[max_time_ind]
       
        print("self.real_path.time[-1]",self.real_path.time[-1])
        print("analytic_dist",analytic_dist)
        print(dist/analytic_dist)
        return dist/analytic_dist

    def update_relative_rewards_and_paths(self):
        self.relative_reward.append(self.comp_relative_reward())
        self.paths.append([self.real_path.velocity,self.real_path.distance])

    def save_run_data(self):
        #try: 
            with open(self.save_run_data_name, 'w') as f:
                for data in self.run_data:
                    f.write("%s \n" % (data))
        #except:
        #    print("cannot save data")      
    def save_readeable_data(self):
        try: 
            with open(self.save_readable_data_name, 'w') as f:
                for i in range(len(self.rewards)):
                    f.write("%s \t %s\t %s \t %s \n" % (i,self.rewards[i],self.relative_reward[i],self.episode_end_mode[i]))
        except:
            print("cannot save data")
    def save_data(self):
        try: 
            with open(self.save_name, 'w') as f:
                json.dump((self.run_num,self.train_num,self.rewards,self.lenght,self.relative_reward, self.episode_end_mode,self.path_seed,self.paths ),f)
            print("data manager saved")            
        except:
            print("cannot save data manager")

    def load_data(self):
        try:
            with open(self.restore_name, 'r') as f:
                self.run_num,self.train_num,self.rewards,self.lenght,self.relative_reward, self.episode_end_mode,self.path_seed,self.paths = json.load(f)#,self.paths
            print("data manager restored")
        except:
            print ("cannot restore data manager:", sys.exc_info()[0])
            #raise
    
