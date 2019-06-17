#import environment1
#import data_manager1
#from hyper_parameters import HyperParameters
#from DDPG_net import DDPG_network
#import numpy as np
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

#import time
#t = time.clock()
#np.array(list(range(100000)))
#print(time.clock() - t)

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




#import matplotlib
#import matplotlib.pyplot as plt
#from matplotlib.colors import BoundaryNorm
#from matplotlib.ticker import MaxNLocator
#import numpy as np

##
##acc, steer = np.mgrid[slice(-1, 1 + da, da),
##                        slice(-1, 1 + da, da)]

##actions = zip(acc, steer)
##a = np.arange(10000).reshape((100, 100))
##print(a)

#da = 0.1
##a = [[i+j for i in np.arange(-1,1+da,da)] for j in np.arange(-1,1+da,da)]

##[0,0],[0,0.1],...,[0,1],[0.1,0],...,[0.1,1],




## make these smaller to increase the resolution
#dx, dy = 0.05, 0.05

## generate 2 2d grids for the x & y bounds


#y, x = np.mgrid[slice(1, 5 + dy, dy),
#                slice(1, 5 + dx, dx)]

#z = np.sin(x)**10 + np.cos(10 + y*x) * np.cos(x)

#da = 0.2
##x = np.arange(-1.0, 1.0+ da, da)
##z = np.sin(x)**10 + np.cos(10 + x*x) * np.cos(x)
## x and y are bounds, so z should be the value *inside* those bounds.
## Therefore, remove the last value from the z array.
##z = z[:-1, :-1]
##levels = MaxNLocator(nbins=15).tick_values(z.min(), z.max())


### pick the desired colormap, sensible levels, and define a normalization
### instance which takes data values and translates those into levels.
#cmap = plt.get_cmap('PiYG')
##norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

#fig, (ax0, ax1) = plt.subplots(nrows=2)

#im = ax0.pcolormesh(x, y, z, cmap=cmap)# norm=norm
##im = ax0.pcolormesh(x, x, z, cmap=cmap)# norm=norm
#fig.colorbar(im, ax=ax0)
#ax0.set_title('pcolormesh with levels')


## contours are *point* based plots, so convert our bound into point
## centers
##cf = ax1.contourf(x[:-1, :-1] + dx/2.,
##                  y[:-1, :-1] + dy/2., z, levels=levels,
##                  cmap=cmap)
##fig.colorbar(cf, ax=ax1)
##ax1.set_title('contourf with levels')

## adjust spacing between subplots so `ax1` title and `ax0` tick labels
## don't overlap
#fig.tight_layout()

#plt.show()

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
# sphinx_gallery_thumbnail_number = 2

vegetables = ["cucumber", "tomato", "lettuce", "asparagus",
              "potato", "wheat", "barley"]
farmers = ["Farmer Joe", "Upland Bros.", "Smith Gardening",
           "Agrifun", "Organiculture", "BioGoods Ltd.", "Cornylee Corp."]

harvest = np.array([[0.8, 2.4, 2.5, 3.9, 0.0, 4.0, 0.0],
                    [2.4, 0.0, 4.0, 1.0, 2.7, 0.0, 0.0],
                    [1.1, 2.4, 0.8, 4.3, 1.9, 4.4, 0.0],
                    [0.6, 0.0, 0.3, 0.0, 3.1, 0.0, 0.0],
                    [0.7, 1.7, 0.6, 2.6, 2.2, 6.2, 0.0],
                    [1.3, 1.2, 0.0, 0.0, 0.0, 3.2, 5.1],
                    [0.1, 2.0, 0.0, 1.4, 0.0, 1.9, 6.3]])


fig, ax = plt.subplots()
im = ax.imshow(harvest,cmap ="YlGn" )

ax.scatter([1],[1],color = 'black')#,'o'
# We want to show all ticks...
#ax.set_xticks(np.arange(len(farmers)))
#ax.set_yticks(np.arange(len(vegetables)))
## ... and label them with the respective list entries
#ax.set_xticklabels(farmers)
#ax.set_yticklabels(vegetables)

## Rotate the tick labels and set their alignment.
#plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
#         rotation_mode="anchor")

## Loop over data dimensions and create text annotations.
#for i in range(len(vegetables)):
#    for j in range(len(farmers)):
#        text = ax.text(j, i, harvest[i, j],
#                       ha="center", va="center", color="w")

#ax.set_title("Harvest of local farmers (in tons/year)")
#fig.tight_layout()
plt.show()