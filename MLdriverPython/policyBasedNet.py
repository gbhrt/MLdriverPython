import tensorflow as tf
import numpy as np
import random
import math
import os
import pathlib
import DQN_net

class Network:
  
    def dynamic_model_estimator(self,action,state,features_num):#define a net - input: state and action - output: next state
        hidden_layer_nodes1 = 50
        hidden_layer_nodes2 = 25

        #linear regression:
        #W1 = tf.Variable(tf.truncated_normal([features_num,action_space_n], stddev=1e-5))
        #b1 = tf.Variable(tf.constant(0.0, shape=[action_space_n]))
        #Q = tf.matmul(state,W1) + b1
        #############################################################
        state_and_action = tf.concat([state,action],1)
        #hidden layer 1:
        W1 = tf.Variable(tf.truncated_normal([features_num+1,hidden_layer_nodes1], stddev=0.1))
        b1 = tf.Variable(tf.constant(0.1, shape=[hidden_layer_nodes1]))
        z1 = tf.nn.relu(tf.add(tf.matmul(state_and_action,W1),b1))
        #hidden layer 2:
        W2 = tf.Variable(tf.truncated_normal([hidden_layer_nodes1,hidden_layer_nodes2], stddev=0.1))
        b2 = tf.Variable(tf.constant(0.1, shape=[hidden_layer_nodes2]))
        z2 = tf.nn.relu(tf.add(tf.matmul(z1,W2),b2))
        #output layer:
        W3 = tf.Variable(tf.truncated_normal([hidden_layer_nodes2,features_num], stddev=0.1))
        b3 = tf.Variable(tf.constant(0.1, shape=[features_num]))
        next_state = tf.add(tf.matmul(z2,W3), b3)
        #############################################################
        return next_state
   


    def actor_critic(self,features_num,action_space_n,alpha_actor = None,alpha_critic = None, tau = 1.0):
        self.state = tf.placeholder(tf.float32, [None,features_num] )
        self.V_ = tf.placeholder(tf.float32,shape=[None])
        self.action = tf.placeholder( dtype=tf.int32, shape=[None] )
        self.actions_one_hot = tf.one_hot(self.action,action_space_n,dtype=tf.float32)#one hot
        self.input_targetQa = tf.placeholder(tf.float32, shape=[None])

        self.Pi = self.policy_estimator(action_space_n, self.state,features_num)
        self.targetPi = self.policy_estimator(action_space_n, self.state,features_num)
        self.Q = self.Q_estimator(action_space_n, self.state,features_num)
        self.targetQ = self.Q_estimator(action_space_n, self.state,features_num)

        #self.V = self.value_estimator(features_num,self.state)
        #self.next_state = self.dynamic_model_estimator(

         
        #self.value_loss = tf.squared_difference(self.V,self.V_)
        #self.update_value = tf.train.GradientDescentOptimizer(alpha_critic).minimize(self.value_loss)
        if alpha_actor !=None and alpha_critic != None:#for training the network
            self.Qa = tf.reduce_sum(tf.multiply(self.Q, self.actions_one_hot),axis=1) #Q on action a at the feeded state
            
            self.Q_loss = tf.squared_difference(self.Qa,self.input_targetQa)
            self.update_Q = tf.train.AdamOptimizer(alpha_critic).minimize(self.Q_loss)
            self.update_target_init(tau)

            self.Pia = tf.reduce_sum(tf.multiply(self.Pi, self.actions_one_hot),axis=1)#Pi on action a at the feeded state

            self.policy_loss = - tf.log(self.Pia+1e-8)*self.V_
            self.update_policy = tf.train.AdamOptimizer(alpha_actor).minimize(self.policy_loss)
            #self.policy_loss = - tf.reduce_mean(tf.log(tf.matmul(self.Pi,tf.transpose(self.actions_one_hot))+1e-8)*self.V_)
            #optimizer = tf.train.AdamOptimizer(alpha_actor)
            #gradients, variables = zip(*optimizer.compute_gradients(self.policy_loss))
            #gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
            #self.update_policy = optimizer.apply_gradients(zip(gradients, variables))

        

        #self.test = -tf.reduce_mean(tf.clip_by_value(tf.log(tf.matmul(self.Pi,tf.transpose(self.action))),-10,10)*self.V_)
        return
    ##################################################################################################################################

    def __init__(self,features_num,action_space_n,alpha_actor = None,alpha_critic = None, tau = 1.0):#
        DQN_net.DQN(self,features_num,action_space_n,alpha_actor,alpha_critic, tau)
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        return


    def get_Pi(self,s):#get one pi at a specific state
        Pi = self.sess.run(self.Pi, feed_dict={self.state: s})
        return Pi
    def get_Q(self,s):#get one pi at a specific state
        Q = self.sess.run(self.Q, feed_dict={self.state: s})
        return Q
    def get_targetQ(self,s):#get one pi at a specific state
        Q = self.sess.run(self.targetQ, feed_dict={self.state: s})
        return Q

    
    def Update_policy(self,s,a,V_):
        self.sess.run(self.update_policy, feed_dict={self.state: s, self.action: a, self.V_: V_})
        return 
    def Update_Q(self,s,a,input_targetQa):
        self.sess.run(self.update_Q, feed_dict={self.state: s ,self.action: a ,self.input_targetQa: input_targetQa})# 
        return 

    def Update_value(self,s,V_):
        self.sess.run(self.update_value, feed_dict={self.state: s ,self.V_: V_})# 
        return 

    def get_par(self):
        return self.sess.run(self.W1),self.sess.run(self.b1), 

    def get_Q_loss(self,s,a,input_targetQa):
        return self.sess.run(self.Q_loss ,feed_dict={self.state: [s], self.input_targetQa: [input_targetQa],self.action:a})
    def get_value_loss(self,s,V_):
        return self.value_loss.eval(session=self.sess,feed_dict={self.state: s, self.V_: V_})
        #return self.sess.run(self.value_loss ,feed_dict={self.state: s, self.V_: V_})
    def get_policy_loss(self,s,action,V_):
        return self.policy_loss.eval(session=self.sess,feed_dict={self.state: s, self.action: action, self.V_: V_})

    
    def update_target_init(self,tau):
        tvars = tf.trainable_variables()
        tvars = [var for var in tvars if 'Q_' in var.name]
        Q_vars = tvars[0:len(tvars)//2]
        tarQ_vars = tvars[len(tvars)//2:len(tvars)]
        self.update_var_vec = []
        for i in range(len(Q_vars)):
            self.update_var_vec.append(tarQ_vars[i].assign((Q_vars[i].value()*tau) + ((1-tau) * tarQ_vars[i])))
        return self.update_var_vec

    def update_target(self):
        for var in self.update_var_vec:
            self.sess.run(var)





