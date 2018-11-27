#import enviroment1
#import data_manager1
#from hyper_parameters import HyperParameters
#from DDPG_net import DDPG_network
#import numpy as np
#import tensorflow as tf
#import time
#def measure():
#    t0 = time.clock()
#    t1 = t0
#    while t1 == t0:
#        t1 = time.clock()
#    return (t0, t1, (t1-t0)*1000000)

#samples = [measure() for i in range(10)]

#for s in samples:
#    print (s)

#def measure_clock():
#    t0 = time.clock()
#    t1 = time.clock()
#    while t1 == t0:
#        t1 = time.clock()
#    return (t0, t1, t1-t0)

#reduce( lambda a,b:a+b, [measure_clock()[2] for i in range(1000000)] )/1000000.0

#HP = HyperParameters()
#dataManager = data_manager1.DataManager(HP.save_file_path,HP.restore_file_path,HP.restore_flag)
#env = enviroment1.OptimalVelocityPlanner(dataManager)
#net = DDPG_network(env.observation_space.shape[0],env.action_space.shape[0],env.action_space.high[0],HP.alpha_actor,HP.alpha_critic,tau = HP.tau,seed = HP.seed)  
#net.restore(HP.restore_file_path)
#state = env.reset()
#state = [0.1 for _ in state]
    
#print("state:",state)
#for vel in range (15):
#    state[0] = float(vel)
#    a = [[0]]
#    Qa = net.get_Qa(np.reshape(state, (1, env.observation_space.shape[0])),a)[0][0]
#    Q0 = net.get_Qa(np.reshape(state, (1, env.observation_space.shape[0])),[[0]])[0][0]
#    Q1 = net.get_Qa(np.reshape(state, (1, env.observation_space.shape[0])),[[1.0]])[0][0]
#    Qneg1 = net.get_Qa(np.reshape(state, (1, env.observation_space.shape[0])),[[-1.0]])[0][0]
#    print("vel:",vel,"Qa:",Qa,"Q0:",Q0,"Q1",Q1,"Qneg1",Qneg1)

#env.close()
import threading
import time
import inspect


lock = threading.Lock()
mainFinish = threading.Event()
threadFinish = threading.Event()

class Thread(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.start()
    def run(self):
        
        while(True):
            mainFinish.wait()
            with lock:
                print("thread")
                time.sleep(0.02)
               




if __name__ == '__main__':   
    hello = Thread()

    while(True):
        print("before lock")
        mainFinish.clear()
        with lock:  
            mainFinish.set()      
            print("after lock")
            print("main")
            
     
        time.sleep(0.5)

        