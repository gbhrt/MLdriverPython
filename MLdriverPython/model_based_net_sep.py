import tensorflow as tf
import tflearn
import os
import pathlib
import numpy as np
import library as lib
import random
from net_lib import NetLib


class model_based_network(NetLib):
    def __init__(self,X_n,Y_n,alpha,net_type = 0):#,dropout_flag = False
        self.X_n = X_n
        self.Y_n = Y_n

        tf.reset_default_graph()   
        #hidden_layer_nodes1 = 200
        #hidden_layer_nodes2 = 100
        hidden_layer_nodes1 = 20
        hidden_layer_nodes2 = 20
        hidden_layer_nodes3 = 20
        hidden_layer_nodes4 = 20

        self.alpha = alpha#0.001

        self.keep_prob = tf.placeholder(tf.float32)
        self.X=tf.placeholder(tf.float32, [None,X_n])

        self.Y_ = []
        for i in range(self.Y_n):
            self.Y_.append(tf.placeholder(tf.float32,[None,1]))

        layer_sizes = [hidden_layer_nodes1,hidden_layer_nodes2,hidden_layer_nodes3]

        self.optimizers = []
        self.Y = []
        self.loss = []
        for i in range(self.Y_n):#create Y_n nets:
            net = self.X
            for layer_size in layer_sizes:
                net = tf.layers.dense(inputs=net, units=layer_size, activation=tf.nn.tanh)
            self.Y.append(tf.layers.dense(inputs=net, units=1))#output from 1 net
            self.loss.append(tf.reduce_mean(tf.squared_difference(self.Y[i],self.Y_[i])))
            self.optimizers.append(tf.train.AdamOptimizer(alpha).minimize(self.loss[i]))


        
        #if net_type == 0:
        #    ##3 hidden layers:
        #    net = tflearn.fully_connected(self.X, hidden_layer_nodes1,regularizer='L2', weight_decay=0.1)
        #    net = tflearn.activations.relu(net)
        #    net = tflearn.fully_connected(net, hidden_layer_nodes2,regularizer='L2', weight_decay=0.1)
        #    net = tflearn.activations.relu(net)
        #    net = tflearn.fully_connected(net, hidden_layer_nodes3,regularizer='L2', weight_decay=0.1)
        #    net = tflearn.activations.relu(net)
        #    #net = tflearn.fully_connected(net, hidden_layer_nodes4,regularizer='L2', weight_decay=0.01)
        #    #net = tflearn.activations.relu(net)
        #    self.Y = tflearn.fully_connected(net, Y_n,regularizer='L2', weight_decay=0.1)
        #    self.loss=tf.reduce_mean(tf.squared_difference(self.Y,self.Y_))
        #else:
        #    layer_sizes = [hidden_layer_nodes1,hidden_layer_nodes2,hidden_layer_nodes3]
        #    layer = self.X
        #    for layer_size in layer_sizes:
        #        layer = tf.layers.dense(inputs=layer, units=layer_size, activation=tf.nn.tanh)
        #    self.Y = tf.layers.dense(inputs=layer, units=1)
        #    self.sigma = tf.layers.dense(inputs=layer, units=1, activation = lambda x: tf.nn.elu(x) + 1)

        #    distribution = tf.distributions.Normal(loc=self.Y, scale=self.sigma)
        #    self.loss = tf.reduce_mean(-distribution.log_prob(self.Y_))

    
        


        

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        print("Network ready")
        return



    #def train(self,num_trains):
    #    batch_size = 64
    #    waitFor = lib.waitFor()#wait for "enter" in another thread - then stop = true
    #    for i in range(num_trains):
    #        if waitFor.stop == [True]:
    #            break
    #        samples = np.array(random.sample(self.train_data,np.clip(batch_size,0,len(self.train_data))))
    #        X = samples[...,0:self.features_num]
    #        Y_ =  samples[...,self.features_num:]
    #        #samples = map(np.array, zip(*samples))
            
    #        self.update_network(X,Y_)
    #        if i %100 == 0:
    #            print(self.get_loss(X,Y_))
    #    self.save_model('models\\',self.file)

    #    return
    #def get_loss_from_data(self,data):
    #    norm_data = lib.normalize(data,self.norm_vec)
    #    X,Y_ = lib.split_to_columns(norm_data,self.features_num)
    #    return self.get_loss(X,Y_)

    


    #def predict(self,sample):
    #    sample_vec = lib.normalize([sample],self.norm_vec[:self.features_num])
    #    prediction = list(self.get_Y(sample_vec)[0]) #self.regressor.predict(sample_vec)#
    #    norm_prediction = lib.denormalize([prediction],self.norm_vec[self.features_num:])[0]
    #    return norm_prediction

        

    def update_network(self,X,Y_,keep_prob = 1.0):
        #self.sess.run(self.update, feed_dict={self.X: lib.normalize(X,self.X_norm_vec) ,self.Y_: Y_})# 
        Y_ = np.array(Y_).transpose().tolist()
        for i in range(self.Y_n):#Y_n
            Y_[i] = [[yi] for yi in Y_[i]]
            self.sess.run(self.optimizers[i], feed_dict={self.X: X ,self.Y_[i]: Y_[i]})
        return 
    def get_loss(self,X,Y_,keep_prob = 1.0):
        #return self.sess.run(self.loss, feed_dict= {self.X: lib.normalize(X,self.X_norm_vec) ,self.Y_: Y_})# 
        Y_ = np.array(Y_).transpose().tolist()
        loss = []
        for i in range(self.Y_n):#Y_n
            Y_[i] = [[yi] for yi in Y_[i]]
            loss.append(self.sess.run(self.loss[i], feed_dict={self.X: X ,self.Y_[i]: Y_[i]}))
        return loss
    #def get_Y_1(self,X):
    #    Y = self.sess.run(self.Y, feed_dict= {self.X: lib.normalize(X,self.X_norm_vec)})
    #    #Y = self.sess.run(self.Y, feed_dict= {self.X: X})
    #    return Y
    def get_Y(self,X,keep_prob = 1.0):
        #norm_X = lib.normalize(X,self.X_norm_vec)
        #print("norm x:",norm_X)
        #Y = self.sess.run(self.Y, feed_dict= {self.X: lib.normalize(X,self.X_norm_vec)})
        Y = []
        for i in range(self.Y_n):#Y_n
            y = self.sess.run(self.Y[i], feed_dict={self.X: X}).flatten().tolist()
            Y.append(y)
        #Y = np.array(self.sess.run(self.Y, feed_dict= {self.X:X}))
        return np.array(Y).transpose()

    def get_Y_sigma(self,X,keep_prob = 1.0):
        Y,sigma = self.sess.run([self.Y,self.sigma], feed_dict= {self.X:X,self.keep_prob: keep_prob})
        return Y,sigma