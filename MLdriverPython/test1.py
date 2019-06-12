#import environment1
#import data_manager1
#from hyper_parameters import HyperParameters
#from DDPG_net import DDPG_network
import numpy as np
#import tensorflow as tf
#import json
#import time
#import collections
#import matplotlib.pyplot as plt
#input_dim=2
#a =[[1,2],[3,4],[5,6]]
#np.array(a)
#a_tr = np.transpose(a)
#print(a_tr)
#a_tr[0] = np.square( a_tr[0] - 1)
#print(a_tr)

import time
t = time.clock()
np.array(list(range(100000)))
print(time.clock() - t)

#model = tf.keras.models.Sequential([
#    tf.keras.layers.Dense(20, activation=tf.keras.activations.relu,input_shape = (2,) ),
#    tf.keras.layers.Dense(20, activation=tf.keras.activations.relu),
#    tf.keras.layers.Dense(20, activation=tf.keras.activations.relu),
#    tf.keras.layers.Dense(1)
#    ])

#model.compile(optimizer=tf.keras.optimizers.Adam(),
#        loss=tf.keras.losses.mean_squared_error,
#        metrics=['accuracy'])

#graph = tf.get_default_graph()

#model.summary()
#print(model.evaluate(np.array([[1,2],[2,4]]),np.array([1,2])))
#model.train_on_batch(np.array([[1,2],[2,4]]),np.array([1,2]))
#print(model.predict(np.array([[1,2],[2,4]]),batch_size = 2))#


#def train():
#    for i in range(100):
#        model.train_on_batch(np.array([[1,2],[2,4]]),np.array([1,2]))
#        print("train model")


#t = threading.Thread(target=train)
#t.start()

#OrderedDict ={'banana': 3, 'apple': 4, 'pear': 1, 'orange': 2}
#x = collections.OrderedDict((("a", "1"), ("c", '3'), ("b", "2")))
#x["d"] = 4
#print(list(x.keys()).index("c"))

#print(x.keys().index("c"))
# importing the required module 
# importing the required modules 
#import timeit 

# binary search function 





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

#import threading
#import time
#import inspect


#lock = threading.Lock()
#mainFinish = threading.Event()
#threadFinish = threading.Event()

#class Thread(threading.Thread):
#    def __init__(self):
#        threading.Thread.__init__(self)
#        self.start()
#    def run(self):
        
#        while(True):
#            mainFinish.wait()
#            with lock:
#                print("thread")
#                time.sleep(0.02)
               




#if __name__ == '__main__':   
#    hello = Thread()

#    while(True):
#        print("before lock")
#        mainFinish.clear()
#        with lock:  
#            mainFinish.set()      
#            print("after lock")
#            print("main")
            
     
#        time.sleep(0.5)
