import tensorflow as tf
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
import random
import math
import os

class Network:
    def __init__(self,alpha,features,action_space_n):#
        
        
        self.hidden_layer_nodes1 = 50
        self.hidden_layer_nodes2 = 25

        self.state = tf.placeholder(tf.float32, [None,features] )
        #self.state_vec = tf.reshape(self.state, [1, features])
        self.Q_ = tf.placeholder(tf.float32, shape=[None, action_space_n])

        #linear regression:
        #self.W1 = tf.Variable(tf.truncated_normal([features,action_space_n], stddev=1e-5))
        #self.b1 = tf.Variable(tf.constant(0.0, shape=[action_space_n]))
        #self.Q = tf.matmul(self.state,self.W1) + self.b1
        #############################################################
        
        #hidden layer 1:
        self.W1 = tf.Variable(tf.truncated_normal([features,self.hidden_layer_nodes1], stddev=0.1))
        self.b1 = tf.Variable(tf.constant(0.1, shape=[self.hidden_layer_nodes1]))
        self.z1 = tf.nn.relu(tf.add(tf.matmul(self.state,self.W1),self.b1))
        #hidden layer 2:
        self.W2 = tf.Variable(tf.truncated_normal([self.hidden_layer_nodes1,self.hidden_layer_nodes2], stddev=0.1))
        self.b2 = tf.Variable(tf.constant(0.1, shape=[self.hidden_layer_nodes2]))
        self.z2 = tf.nn.relu(tf.add(tf.matmul(self.z1,self.W2),self.b2))
        #output layer:
        self.W3 = tf.Variable(tf.truncated_normal([self.hidden_layer_nodes2,action_space_n], stddev=0.1))
        self.b3 = tf.Variable(tf.constant(0.1, shape=[action_space_n]))
        self.Q = tf.add(tf.matmul(self.z2,self.W3), self.b3)
        #############################################################

        
        self.loss = tf.pow(self.Q - self.Q_, 2)
        self.update = tf.train.GradientDescentOptimizer(alpha).minimize(self.loss)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

       ##########################initilize weights
        #self.sess.run(tf.assign(self.W1, [[1,-1],[1,-1]]))
        #self.sess.run(tf.assign(self.b1,[0,0]))
        #self.sess.run(tf.assign(self.W1, [[-1,0,1],[1,0,-1]]))
        #self.sess.run(tf.assign(self.b1,[0,0,0]))
        #l = int((action_space_n)/2)
        #self.sess.run(tf.assign(self.W1, [[x for x in range(-l,0)]+[x for x in range(0,l+1)],[x for x in range(0,l+1)]+[x for x in range(-l,0)]]))
        #self.sess.run(tf.assign(self.b1,[0 for _ in range(0,2*l +1)]))
      #########################
        return

    def get_Q(self,s):#get one Q at a specific state
        Q = self.sess.run(self.Q, feed_dict={self.state: [s]})
        return Q[0]

    def update_sess(self,s,Q_corrected):
        self.sess.run(self.update, feed_dict={self.state: s, self.Q_: Q_corrected})
        return

    def save_model(self,*args):
        
         path = os.getcwd()
         path += "\models\ "
         
         #path = "C:\MachineLearning\MLdriverPython\MLdriverPython\models\ "
         if len(args) > 0:
             file_name = path+args[0]
         else:
            file_name =  path+"model3.ckpt" #/media/windows-share/MLdriverPython/MLdriverPython/
         saver = tf.train.Saver()
         save_path = saver.save(self.sess, file_name)
         print("Model saved in file: %s" % save_path)

    def restore(self,*args):
        path = os.getcwd()
        path += "\models\ "
        #path = "C:\MachineLearning\MLdriverPython\MLdriverPython\models\ "
        if args:
            file_name = path+args[0]
        else:
            file_name = path+"model3.ckpt"#/media/windows-share/MLdriverPython/MLdriverPython/
       
        
            # Restore variables from disk.
        saver = tf.train.Saver()
        saver.restore(self.sess, file_name)
        print("Model restored.")

    def get_par(self):
        return self.sess.run(self.W1),self.sess.run(self.b1), 

    def get_loss(self,s,Q_corrected):
        return self.sess.run(self.loss ,feed_dict={self.state: [s], self.Q_: [Q_corrected]})





