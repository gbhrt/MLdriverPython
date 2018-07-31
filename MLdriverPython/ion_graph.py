import numpy as np
import matplotlib.pyplot as plt

class ionGraph:
    def __init__(self):
        plt.ion()
    def update(self,guiShared):
        if guiShared.state is not None:
            plt.plot(guiShared.state['roll'])
            plt.pause(0.1) 
