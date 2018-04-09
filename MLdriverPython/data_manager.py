import plot
import classes
import numpy as np
import json
class DataManager:
    def __init__(self,file = None):
        self.real_path = classes.Path()
        self.features = []
        self.rewards = []
        self.ind = 0
        self.acc = []
        self.dec = []
        self.data = []
        if file != None:
            self.file_name = file
        
        #plt_lib = plot.Plot()

    def save_data(self):
        with open(self.file_name, 'w') as f:
            json.dump(self.data,f)
    def load_data(self):
        with open(self.file_name, 'r') as f:
            self.data = json.load(f)


    def comp_rewards(self,path_num,gamma):
        total_reward = 0
        for i,r in enumerate(self.rewards):
            total_reward+=r*gamma**i
        mean_reward = sum(self.rewards)/len(self.rewards)
        dat = [total_reward,mean_reward]
        if len(self.data) < path_num:
            self.data.append([dat])
        else:
            self.data[path_num].append(dat)
        self.save_data()
    def print_data(self):
        for path_num,dat in enumerate(self.data):
            print(path_num)
            np_dat = np.array(dat)
            print("total reward: ",np_dat[:,0])
            print("mean reward: ",np_dat[:,1])
    def restart(self):
        self.real_path = classes.Path()
        self.features = []
        self.rewards = []
        self.acc = []
        self.dec = []
        self.ind = 0
    def save_additional_data(self,pl = None,features = None, action = None,reward = None):
        if reward != None:
            self.rewards.append(reward)
        if features != None:
            vel = []
            #for i in range(len(features)):
            #    vel.append(pl.simulator.vehicle.velocity - features[i] )
            self.features.append(features)
        if action != None:
            dist = pl.in_vehicle_reference_path.distance[pl.main_index]
            if action > 0:
                self.acc.append([self.ind,dist])
            else:
                self.dec.append([self.ind,dist])
            self.ind += 1
        return
    def update_real_path(self,pl, velocity_limit = None):

        dist = pl.in_vehicle_reference_path.distance[pl.main_index]
        self.real_path.distance.append(dist)
        self.real_path.position.append(pl.simulator.vehicle.position)
        self.real_path.velocity.append(pl.simulator.vehicle.velocity)
        self.real_path.time.append(pl.get_time())
        if velocity_limit != None: self.real_path.velocity_limit.append(velocity_limit)
 
        return