#import enviroment1
#import data_manager1
#from hyper_parameters import HyperParameters
#from DDPG_net import DDPG_network
#import numpy as np
#import tensorflow as tf
#import time
import collections

#OrderedDict ={'banana': 3, 'apple': 4, 'pear': 1, 'orange': 2}
x = collections.OrderedDict((("a", "1"), ("c", '3'), ("b", "2")))
x["d"] = 4
print(list(x.keys()).index("c"))

#print(x.keys().index("c"))


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

        