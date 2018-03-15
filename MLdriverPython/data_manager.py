import plot
import classes
import numpy as np
class DataManager:
    def __init__(self):
        self.real_path = classes.Path()
        self.features = []
        self.ind = 0
        self.acc = []
        self.dec = []
        self.data = []
        
        #plt_lib = plot.Plot()
    
    def comp_rewards(self,path_num,episode_rewards,gamma):
        total_reward = 0
        for i,r in enumerate(episode_rewards):
            total_reward+=r*gamma**i
        mean_reward = sum(episode_rewards)/len(episode_rewards)
        dat = np.array([total_reward,mean_reward])
        if len(self.data) < path_num:
            self.data.append(np.array([dat]))
        else:
            self.data[path_num].append(dat)
    def print_data(self):
        for path_num,dat in enumerate(self.data):
            print(path_num)
            print("total reward: ",dat[:,0])
            print("mean reward: ",dat[:,1])
    def restart(self):
        self.real_path = classes.Path()
        self.features = []
        self.acc = []
        self.dec = []
        self.ind = 0
    def save_additional_data(self,pl,features = None, action = None):
        if features != None:
            vel = []
            for i in range(len(features)):
                vel.append(pl.simulator.vehicle.velocity - features[i] )
            self.features.append(vel)
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