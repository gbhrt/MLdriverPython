import tensorflow as tf
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
import random
import math
import os


class Network:

    def policy_estimator(self,features,action_space_n,state): #define a net - input: state (and dimentions) - output: Pi - policy
        hidden_layer_nodes1 = 50
        hidden_layer_nodes2 = 25

        #linear regression:
        #theta1 = tf.Variable(tf.truncated_normal([features,action_space_n], stddev=1e-5))
        #theta_b1 = tf.Variable(tf.constant(0.0, shape=[action_space_n]))
        #pi = tf.nn.softmax(tf.matmul(state,theta1) + theta_b1)
        #############################################################
        
        #hidden layer 1:
        theta1 = tf.Variable(tf.truncated_normal([features,hidden_layer_nodes1], stddev=0.1))
        theta_b1 = tf.Variable(tf.constant(0.1, shape=[hidden_layer_nodes1]))
        theta_z1 = tf.nn.relu(tf.add(tf.matmul(state,theta1),theta_b1))
        #hidden layer 2:
        theta2 = tf.Variable(tf.truncated_normal([hidden_layer_nodes1,hidden_layer_nodes2], stddev=0.1))
        theta_b2 = tf.Variable(tf.constant(0.1, shape=[hidden_layer_nodes2]))
        theta_z2 = tf.nn.relu(tf.add(tf.matmul(theta_z1,theta2),theta_b2))
        #output layer:
        theta3 = tf.Variable(tf.truncated_normal([hidden_layer_nodes2,action_space_n], stddev=0.1))
        theta_b3 = tf.Variable(tf.constant(0.1, shape=[action_space_n]))
        pi = tf.add(tf.matmul(theta_z2,theta3), theta_b3)
        pi = tf.nn.softmax(pi)
        #############################################################
        return pi
    def Q_estimator(self,features,action_space_n,state):#define a net - input: state (and dimentions) - output: Q - Value
        hidden_layer_nodes1 = 50
        hidden_layer_nodes2 = 25

        #linear regression:
        W1 = tf.Variable(tf.truncated_normal([features,action_space_n], stddev=1e-5))
        b1 = tf.Variable(tf.constant(0.0, shape=[action_space_n]))
        Q = tf.matmul(state,W1) + b1
        #############################################################
        
        ##hidden layer 1:
        #W1 = tf.Variable(tf.truncated_normal([features,hidden_layer_nodes1], stddev=0.1))
        #b1 = tf.Variable(tf.constant(0.1, shape=[hidden_layer_nodes1]))
        #z1 = tf.nn.relu(tf.add(tf.matmul(state,W1),b1))
        ##hidden layer 2:
        #W2 = tf.Variable(tf.truncated_normal([hidden_layer_nodes1,hidden_layer_nodes2], stddev=0.1))
        #b2 = tf.Variable(tf.constant(0.1, shape=[hidden_layer_nodes2]))
        #z2 = tf.nn.relu(tf.add(tf.matmul(z1,W2),b2))
        ##output layer:
        #W3 = tf.Variable(tf.truncated_normal([hidden_layer_nodes2,action_space_n], stddev=0.1))
        #b3 = tf.Variable(tf.constant(0.1, shape=[action_space_n]))
        #Q = tf.add(tf.matmul(z2,W3), b3)
        #############################################################
        return Q
    def value_estimator(self,features,state):#define a net - input: state (and dimentions) - output: Q - Value
        hidden_layer_nodes1 = 50
        hidden_layer_nodes2 = 25
        output_nodes = 1
        #linear regression:
        #W1 = tf.Variable(tf.constant(0.0, shape=[features,output_nodes]))#tf.truncated_normal([features,1], stddev=1e-5))
        #b1 = tf.Variable(tf.constant(0.0, shape=[output_nodes]))
        #V = tf.matmul(state,W1) + b1

        #############################################################
        
        #hidden layer 1:
        W1 = tf.Variable(tf.truncated_normal([features,hidden_layer_nodes1], stddev=0.1))
        b1 = tf.Variable(tf.constant(0.1, shape=[hidden_layer_nodes1]))
        z1 = tf.nn.relu(tf.add(tf.matmul(state,W1),b1))
        #hidden layer 2:
        W2 = tf.Variable(tf.truncated_normal([hidden_layer_nodes1,hidden_layer_nodes2], stddev=0.1))
        b2 = tf.Variable(tf.constant(0.1, shape=[hidden_layer_nodes2]))
        z2 = tf.nn.relu(tf.add(tf.matmul(z1,W2),b2))
        #output layer:
        W3 = tf.Variable(tf.truncated_normal([hidden_layer_nodes2,output_nodes], stddev=0.1))
        b3 = tf.Variable(tf.constant(0.1, shape=[output_nodes]))
        V = tf.add(tf.matmul(z2,W3), b3)
        #############################################################
        return V

    def __init__(self,alpha_actor,alpha_critic,features,action_space_n):#

        self.state = tf.placeholder(tf.float32, [None,features] )

        self.Pi = self.policy_estimator(features,action_space_n, self.state)
        self.Q = self.Q_estimator(features,action_space_n, self.state)
        self.V = self.value_estimator(features,self.state)

        self.V_ = tf.placeholder(tf.float32,shape=[None, 1])
        self.action = tf.placeholder( tf.float32, [None,2] )
        self.Q_ = tf.placeholder(tf.float32, shape=[None, action_space_n])

        self.value_loss = tf.squared_difference(self.V,self.V_)
        self.update_value = tf.train.GradientDescentOptimizer(alpha_critic).minimize(self.value_loss)

        self.Q_loss = tf.squared_difference(self.Q,self.Q_)
        self.update_Q = tf.train.GradientDescentOptimizer(alpha_critic).minimize(self.Q_loss)
        
        self.policy_loss = - tf.reduce_mean(tf.log(tf.matmul(self.Pi,tf.transpose(self.action))+1e-8)*self.V_)# *self.action
        self.update_policy = tf.train.AdamOptimizer(alpha_actor).minimize(self.policy_loss)

        self.test = -tf.reduce_mean(tf.clip_by_value(tf.log(tf.matmul(self.Pi,tf.transpose(self.action))),-10,10)*self.V_)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

       #########################initilize weights
        #self.sess.run(tf.assign(self.W1, [[1,-1],[-1,1]]))
        #self.sess.run(tf.assign(self.b1,[0,0]))
        #self.sess.run(tf.assign(self.W1, [[-1,0,1],[1,0,-1]]))
        #self.sess.run(tf.assign(self.b1,[0,0,0]))
        #l = int((action_space_n)/2)
        #self.sess.run(tf.assign(self.W1, [[x for x in range(-l,0)]+[x for x in range(0,l+1)],[x for x in range(0,l+1)]+[x for x in range(-l,0)]]))
        #self.sess.run(tf.assign(self.b1,[0 for _ in range(0,2*l +1)]))
      ########################
        return

    
    def get_test(self):
        return self.sess.run(self.test, feed_dict={self.Pi: [[1 - 1e-50,1e-50]],self.action: [[0,1]],self.V_: [[10]]})

    def get_Pi(self,s):#get one pi at a specific state
        Pi = self.sess.run(self.Pi, feed_dict={self.state: [s]})
        return Pi[0]
    def get_Q(self,s):#get one pi at a specific state
        Q = self.sess.run(self.Q, feed_dict={self.state: [s]})
        return Q[0]
    def get_V(self,s):#get one pi at a specific state
        V = self.sess.run(self.V, feed_dict={self.state: [s]})
        return V[0]
    #def update_sess(self,s,pi_):
    #    self.sess.run(self.update, feed_dict={self.state: s, self.pi_: pi_})
    #    return
    
    def Update_policy(self,s,action,V_):
        self.sess.run(self.update_policy, feed_dict={self.state: s, self.action: action, self.V_: V_})
        return 
    def Update_Q(self,s,Q_):
        self.sess.run(self.update_Q, feed_dict={self.state: s ,self.Q_: Q_})# 
        return 
    def Update_value(self,s,V_):
        self.sess.run(self.update_value, feed_dict={self.state: s ,self.V_: V_})# 
        return 

    def get_par(self):
        return self.sess.run(self.W1),self.sess.run(self.b1), 

    def get_Q_loss(self,s,Q_):
        return self.sess.run(self.Q_loss ,feed_dict={self.state: [s], self.Q_: [Q_]})
    def get_value_loss(self,s,V_):
        return self.value_loss.eval(session=self.sess,feed_dict={self.state: s, self.V_: V_})
        #return self.sess.run(self.value_loss ,feed_dict={self.state: s, self.V_: V_})
    def get_policy_loss(self,s,action,V_):
        return self.policy_loss.eval(session=self.sess,feed_dict={self.state: s, self.action: action, self.V_: V_})


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




