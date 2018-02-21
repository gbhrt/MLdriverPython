import matplotlib.pyplot as plt
import numpy as np


class Plot():
    def __init__(self):
        self.fig = 0
        self.ax = 0
    def plot_path(self,path):
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(211)
        self.ax1 = self.fig.add_subplot(212)
        #datax = [x for x in range(9)]
        #datay = [x for x in range(5)]
        #datax = np.arange(0, 10, 0.2)
        #datay = np.sin(datax)
        datax1 = np.array(path.distance)
        datax = np.array(path.time)
        datay = np.array(path.velocity)
        #print("time: ",path.time)
        #print("velocity: ",path.velocity)
        #plt.ion()
        #self.ax1.plot(datax1, datay)
        self.ax.plot(datax, datay)
        plt.show(block = False)
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
