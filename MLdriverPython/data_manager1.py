import plot
import classes
import numpy as np
import json
import plot_lib 
import matplotlib.pyplot as plt
import numpy as np
import pathlib
import library as lib
class DataManager():
    def __init__(self,save_path = None, restore_path = None,restore_flag = False):
        if save_path != None:
            save_path += "data_manager\\"
            pathlib.Path(save_path).mkdir(parents=True, exist_ok=True) 
            self.save_name = save_path+"data.txt"
        else:
            self.save_name ="data.txt"
        if restore_path != None:
            self.restore_name = restore_path+"data_manager\\data.txt"
        else:
            self.restore_name = "data.txt"
        #define data variables 
        self.real_path = classes.Path()
        self.rewards = []
        self.lenght = []
        self.relative_reward = []
        self.episode_end_mode = []
        #reset every episode:
        self.Qa = []
        self.Q0 = []
        self.Q1 = []
        self.Qneg1 = []

        ###############################
        
        if restore_flag:
            self.load_data()

    def plot_all(self):
        plt.figure(1,figsize = (10,5))
        plt.subplot(221) 
        #print("plot all")
        plot_lib.plot_path(self.real_path)

        #plt.subplot(222)  
        #x = np.array(self.real_path.distance)
        #Qa, = plt.plot(x,self.Qa,label = "Qa")
        #Q0, = plt.plot(x,self.Q0,label = "Q0")
        #Q1, = plt.plot(x,self.Q1,label = "Q1")
        #Qneg1, = plt.plot(x,self.Qneg1,label = "Qneg1")
        #plt.legend(handles=[Qa, Q0,Q1,Qneg1])

        plt.subplot(223)  
        plt.title("episodes reward")
        plt.plot(self.rewards)
        plt.plot(lib.running_average(self.rewards,5))
        plt.plot(lib.running_average(self.rewards,100))
        
        plt.subplot(224)  
        plt.title("relative reward")
        plt.plot(self.relative_reward)
        #plt.plot(lib.running_average(self.relative_reward,5))

        #plt.title("episodes lenght")
        #plt.plot(self.lenght)



        plt.show()

        return

    def update_real_path(self,pl, velocity_limit = None, analytic_vel = None,curvature = None):

        dist = pl.in_vehicle_reference_path.distance[pl.main_index]
        self.real_path.distance.append(dist)
        self.real_path.position.append(pl.simulator.vehicle.position)
        self.real_path.velocity.append(pl.simulator.vehicle.velocity)
        self.real_path.time.append(pl.get_time())
        if velocity_limit != None: self.real_path.velocity_limit.append(velocity_limit)
        if analytic_vel != None: self.real_path.analytic_velocity.append(analytic_vel)
        if curvature != None: self.real_path.curvature.append(curvature*50.0)
        return

    def restart(self):
        self.real_path = classes.Path()
        self.Qa = []
        self.Q0 = []
        self.Q1 = []
        self.Qneg1 = []
        

    def save_data(self):
        try: 
            with open(self.save_name, 'w') as f:
                json.dump((self.rewards,self.lenght,self.relative_reward, self.episode_end_mode ),f)
            print("data manager saved")            
        except:
            print("cannot save data manager")

    def load_data(self):
        try:
            with open(self.restore_name, 'r') as f:
                self.rewards,self.lenght,self.relative_reward, self.episode_end_mode = json.load(f)#
            print("data manager restored")
        except:
            print("cannot restore data manager")
