import matplotlib.pyplot as plt
import numpy as np


class Plot():
    def __init__(self):
        self.fig = 0
        self.ax = 0
    def plot_path_with_features(self,data,distance_between_points):
        #while True:
            #inp = input("insert index: ")
            #if inp == 'e':
            #    break
            #index = int(inp)
            index = 0
            if index < len(data.real_path.distance):
                plt.plot(np.array(data.real_path.distance),np.array(data.real_path.velocity))
                plt.plot(np.array(data.real_path.distance),np.array(data.real_path.velocity_limit))
                x = [x*distance_between_points+data.real_path.distance[index] for x in range(len(data.features[index]))]
                plt.plot(np.array(x),np.array(data.features[index]),'ro')
                ac = np.array(data.acc)
                x = ac[:,1]
                y = []
                for i in range( len(data.acc)):
                    y.append(data.real_path.velocity[data.acc[i][0]])
                #plt.plot(x,np.array(y))
                plt.scatter(x,np.array(y),c = "g")
                dc = np.array(data.dec)
                x = dc[:,1]
                y = []
                for i in range( len(data.dec)):
                    y.append(data.real_path.velocity[data.dec[i][0]])
                #plt.plot(x,np.array(y))
                plt.scatter(x,np.array(y),c = "r")
                plt.show()
   
    def plot_path(self,path, block = False):
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)#211
        #self.ax1 = self.fig.add_subplot(212)
        #datax = [x for x in range(9)]
        #datay = [x for x in range(5)]
        #datax = np.arange(0, 10, 0.2)
        #datay = np.sin(datax)
        #datax1 = np.array(path.distance)
        datax = np.array(path.time)
        datay1 = np.array(path.velocity)
        datay2 = np.array(path.velocity_limit)
        #print("time: ",path.time)
        #print("velocity: ",path.velocity)
        #plt.ion()
        #self.ax1.plot(datax1, datay)
        self.ax.plot(datax, datay1)
        self.ax.plot(datax, datay2)
        plt.show(block = block)
        #plt.draw()
        return
    def close(self):
        plt.close("all")
    def start(self):
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)
        plt.ion()

    def update(self,path):
        self.ax.scatter(path.time[-1], path.velocity[-1])
        plt.pause(0.05)
    def end(self):
        plt.ioff()
        self.fig.show()
