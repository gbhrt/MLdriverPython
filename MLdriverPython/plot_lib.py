import matplotlib.pyplot as plt
import numpy as np


#def compare_paths(planned_path):

def plot_path(path):
    

    datax = np.array(path.distance)
    #datax = np.array(path.time)
    datay1 = np.array(path.velocity)
    datay2 = np.array(path.analytic_velocity_limit)
    datay3 = np.array(path.analytic_velocity)
    datay4 = np.array(path.curvature)

    real_vel, = plt.plot(datax, datay1,label = "vehicle velocity")
    vel_lim, = plt.plot(datax, datay2,label = "velocity limit")
    vel, = plt.plot(datax, datay3, label = "analytic velocity")
    #curv, = plt.plot(datax, datay4, label = "curvature")
    plt.legend(handles=[real_vel, vel_lim,vel])#,curv
    #plt.show()

    return


