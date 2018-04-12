import plot
import _thread
import numpy as np
import matplotlib.pyplot as plt
import time

def plot_loop(plot_data):
    while True:
        plt.plot(plot_data)
        plt.show(block = False)
        time.sleep(0.1)

d1 = np.array([1,2,3])
d2 = np.array([3,2,3])
plot_data = d1
print (plot_data)
_thread.start_new_thread(plot_loop,(plot_data,))

for i in range(100):
    plot_data = d2
    time.sleep(1.0)
    plot_data = d1
    time.sleep(1.0)