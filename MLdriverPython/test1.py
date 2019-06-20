#import environment1
#import data_manager1
#from hyper_parameters import HyperParameters
#from DDPG_net import DDPG_network
#import numpy as np
import tensorflow as tf
import os
import pathlib
#import json
#import time
#import collections
#import matplotlib.pyplot as plt


#import time


tf.reset_default_graph()  
graph1 = tf.get_default_graph()
v1 = tf.get_variable("v1", shape=[3], initializer = tf.zeros_initializer)

sess1 = tf.Session()
sess1.run(tf.global_variables_initializer())




tf.reset_default_graph()  
graph2 = tf.get_default_graph()
v1 = tf.get_variable("v1", shape=[3], initializer = tf.zeros_initializer)

sess2 = tf.Session()
sess2.run(tf.global_variables_initializer())



with graph1.as_default():
    saver = tf.train.Saver()
    name = 'tfmodel'
    path = os.getcwd() + "\\files\\models\\"
    path1 = path+name+"\\"
    pathlib.Path(path1).mkdir(parents=True, exist_ok=True) 
    file_name =  path1+name+".ckpt "
    save_path = saver.save(sess1, file_name)
    print("Model saved in file: %s" % save_path)

with graph2.as_default():
    saver = tf.train.Saver()
    name = 'tfmodel1'
    path = os.getcwd() + "\\files\\models\\"
    path1 = path+name+"\\"
    pathlib.Path(path1).mkdir(parents=True, exist_ok=True) 
    file_name =  path1+name+".ckpt "
    save_path = saver.save(sess2, file_name)
    print("Model saved in file: %s" % save_path)



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


