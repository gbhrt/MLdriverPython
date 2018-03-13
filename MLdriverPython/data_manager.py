import plot
import classes
class DataManager:
    def __init__(self):
        self.real_path = classes.Path()
        self.features = []
        self.ind = 0
        self.acc = []
        self.dec = []
        #plt_lib = plot.Plot()
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