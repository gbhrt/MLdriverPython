import tensorflow as tf
import tflearn
import os
import pathlib
import numpy as np
import library as lib
import random
from net_lib import NetLib

from tensorflow.python.client import timeline


class model_based_network(NetLib):
    def __init__(self,X_n,Y_n,alpha,net_type = 0):#,dropout_flag = False

        tf.reset_default_graph()   
        #hidden_layer_nodes1 = 200
        #hidden_layer_nodes2 = 100
        hidden_layer_nodes1 = 100
        hidden_layer_nodes2 = 100
        hidden_layer_nodes3 = 100
        hidden_layer_nodes4 = 100

        self.alpha = alpha#0.001

        self.keep_prob = tf.placeholder(tf.float32)
        self.X=tf.placeholder(tf.float32, [None,X_n])
        self.Y_=tf.placeholder(tf.float32,[None,Y_n])

        #self.W1 = tf.Variable(tf.truncated_normal([X_n,hidden_layer_nodes1], stddev=0.1))
        #self.b1 = tf.Variable(tf.constant(0.1, shape=[hidden_layer_nodes1]))
        #self.z1 = tf.nn.relu(tf.add(tf.matmul(self.X,self.W1),self.b1))

        #self.W2 = tf.Variable(tf.truncated_normal([hidden_layer_nodes1,hidden_layer_nodes2], stddev=0.1))
        #self.b2 = tf.Variable(tf.constant(0.1, shape=[hidden_layer_nodes2]))
        #self.z2 = tf.nn.relu(tf.matmul(self.z1,self.W2)+self.b2)

        #self.W3 = tf.Variable(tf.truncated_normal([hidden_layer_nodes2,Y_n], stddev=0.1))
        #self.b3 = tf.Variable(tf.constant(0.1, shape=[Y_n]))
        #self.Y = tf.matmul(self.z2,self.W3)+self.b3

        #init = tflearn.initializations.truncated_normal(stddev = 0.1)
        #fc1 = tflearn.fully_connected(self.X, hidden_layer_nodes1,weights_init = init,bias_init = tf.constant(0.1, shape=[hidden_layer_nodes1]))
        #fc2 = tflearn.fully_connected(fc1, hidden_layer_nodes2,weights_init = init,bias_init = tf.constant(0.1, shape=[hidden_layer_nodes2]))
        #self.Y = tflearn.fully_connected(fc2, Y_n,weights_init = init,bias_init = tf.constant(0.1, shape=[Y_n]))

        #2 hidden layers:
        #hidden_layer_nodes1 = 400
        #hidden_layer_nodes2 = 300
        #fc1 = tflearn.fully_connected(self.X, hidden_layer_nodes1,regularizer='L2', weight_decay=0.01)
        #fc1 = tflearn.activations.relu(fc1)
        #fc2 = tflearn.fully_connected(fc1, hidden_layer_nodes2,regularizer='L2', weight_decay=0.01)
        #fc2 = tflearn.activations.relu(fc2)
        #self.Y = tflearn.fully_connected(fc2, Y_n,regularizer='L2', weight_decay=0.01)
        if net_type == 0:
            ##3 hidden layers:
            net = tflearn.fully_connected(self.X, hidden_layer_nodes1,regularizer='L2', weight_decay=0.1)
            net = tflearn.activations.relu(net)
            net = tflearn.fully_connected(net, hidden_layer_nodes2,regularizer='L2', weight_decay=0.1)
            net = tflearn.activations.relu(net)
            net = tflearn.fully_connected(net, hidden_layer_nodes3,regularizer='L2', weight_decay=0.1)
            net = tflearn.activations.relu(net)
            #net = tflearn.fully_connected(net, hidden_layer_nodes4,regularizer='L2', weight_decay=0.01)
            #net = tflearn.activations.relu(net)
            self.Y = tflearn.fully_connected(net, Y_n,regularizer='L2', weight_decay=0.1)
            self.loss=tf.reduce_mean(tf.squared_difference(self.Y,self.Y_))
        else:
            layer_sizes = [hidden_layer_nodes1,hidden_layer_nodes2,hidden_layer_nodes3]
            layer = self.X
            for layer_size in layer_sizes:
                layer = tf.layers.dense(inputs=layer, units=layer_size, activation=tf.nn.tanh)
            self.Y = tf.layers.dense(inputs=layer, units=1)
            self.sigma = tf.layers.dense(inputs=layer, units=1, activation = lambda x: tf.nn.elu(x) + 1)

            distribution = tf.distributions.Normal(loc=self.Y, scale=self.sigma)
            self.loss = tf.reduce_mean(-distribution.log_prob(self.Y_))

            #net = tflearn.fully_connected(self.X, hidden_layer_nodes1,regularizer='L2', weight_decay=0.1)
            #net = tflearn.activations.relu(net)
            #net = tflearn.fully_connected(net, hidden_layer_nodes2,regularizer='L2', weight_decay=0.1)
            #net = tflearn.activations.relu(net)
            #net = tflearn.fully_connected(net, hidden_layer_nodes3,regularizer='L2', weight_decay=0.1)
            #net = tflearn.activations.relu(net)
            ##net = tflearn.fully_connected(net, hidden_layer_nodes4,regularizer='L2', weight_decay=0.01)
            ##net = tflearn.activations.relu(net)
            #self.Y = tflearn.fully_connected(net, Y_n,regularizer='L2', weight_decay=0.1)
            ##self.loss=tf.reduce_mean(abs(self.Y-self.Y_))
            #var = 2
            #cost = 10
            
            #zero_range = tf.count_nonzero(tf.keras.backend.less(self.Y-self.Y_,var) and tf.keras.backend.less(-var,self.Y-self.Y_))
            #print("zero_range:",zero_range)
            #self.loss = zero_range * (var*cost) + (1-zero_range)*(self.Y-self.Y_ + var*cost)
            #self.loss = (1-zero_range)* tf.reduce_mean(abs(self.Y-self.Y_))
        #else:
        #    #dropout:

        #    net = tflearn.fully_connected(self.X, hidden_layer_nodes1)#,regularizer='L2', weight_decay=0.1)
        #    net = tflearn.activations.relu(net)
        #    net = tf.nn.dropout (net, self.keep_prob, noise_shape=None)
        #    net = tflearn.fully_connected(net, hidden_layer_nodes2)#,regularizer='L2', weight_decay=0.1)
        #    net = tflearn.activations.relu(net)
        #    net = tf.nn.dropout (net, self.keep_prob, noise_shape=None)
        #    net = tflearn.fully_connected(net, hidden_layer_nodes3)#,regularizer='L2', weight_decay=0.1)
        #    net = tflearn.activations.relu(net)
        #    #net = tflearn.fully_connected(net, hidden_layer_nodes4,regularizer='L2', weight_decay=0.01)
        #    #net = tflearn.activations.relu(net)
        #    net = tf.nn.dropout (net, self.keep_prob, noise_shape=None)
        #    self.Y = tflearn.fully_connected(net, Y_n)#,regularizer='L2', weight_decay=0.1)

        #init = tflearn.initializations.truncated_normal(stddev = 0.1)

        #net = tflearn.fully_connected(self.X, hidden_layer_nodes1,regularizer='L2', weight_decay=0.1,weights_init = init,bias_init =init)
        #net = tflearn.activations.relu(net)
        #net = tf.nn.dropout (net, self.keep_prob, noise_shape=None)
        #net = tflearn.fully_connected(net, hidden_layer_nodes2,regularizer='L2', weight_decay=0.1,weights_init = init,bias_init =init)
        #net = tflearn.activations.relu(net)
        #net = tf.nn.dropout (net, self.keep_prob, noise_shape=None)
        #net = tflearn.fully_connected(net, hidden_layer_nodes3,regularizer='L2', weight_decay=0.1,weights_init = init,bias_init =init)
        #net = tflearn.activations.relu(net)
        ##net = tflearn.fully_connected(net, hidden_layer_nodes4,regularizer='L2', weight_decay=0.01)
        ##net = tflearn.activations.relu(net)
        #net = tf.nn.dropout (net, self.keep_prob, noise_shape=None)
        #self.Y = tflearn.fully_connected(net, Y_n,regularizer='L2', weight_decay=0.1)
        


        self.update=tf.train.AdamOptimizer(alpha).minimize(self.loss)

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

    def check_time():
        run_metadata = tf.RunMetadata()
        self.sess.run(self.Y, feed_dict= {self.X:X,self.keep_prob: keep_prob},run_metadata=run_metadata)
        fetched_timeline = timeline.Timeline(run_metadata.step_stats)
        chrome_trace = fetched_timeline.generate_chrome_trace_format()
        with open('timeline_02_step_%d.json' % i, 'w') as f:
            f.write(chrome_trace)

    def update_network(self,X,Y_,keep_prob = 1.0):
        #self.sess.run(self.update, feed_dict={self.X: lib.normalize(X,self.X_norm_vec) ,self.Y_: Y_})# 
        self.sess.run(self.update, feed_dict={self.X: X ,self.Y_: Y_,self.keep_prob: keep_prob})
        return 
    def get_loss(self,X,Y_,keep_prob = 1.0):
        #return self.sess.run(self.loss, feed_dict= {self.X: lib.normalize(X,self.X_norm_vec) ,self.Y_: Y_})# 
        return self.sess.run(self.loss, feed_dict= {self.X: X ,self.Y_: Y_,self.keep_prob: keep_prob})# 
    #def get_Y_1(self,X):
    #    Y = self.sess.run(self.Y, feed_dict= {self.X: lib.normalize(X,self.X_norm_vec)})
    #    #Y = self.sess.run(self.Y, feed_dict= {self.X: X})
    #    return Y
    def get_Y(self,X,keep_prob = 1.0):
        #norm_X = lib.normalize(X,self.X_norm_vec)
        #print("norm x:",norm_X)
        #Y = self.sess.run(self.Y, feed_dict= {self.X: lib.normalize(X,self.X_norm_vec)})
        Y = self.sess.run(self.Y, feed_dict= {self.X:X,self.keep_prob: keep_prob})
        return Y
    def get_Y_sigma(self,X,keep_prob = 1.0):
        Y,sigma = self.sess.run([self.Y,self.sigma], feed_dict= {self.X:X,self.keep_prob: keep_prob})
        return Y,sigma