import tensorflow as tf
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
import random
import math


class Network:
    def __init__(self,features,action_space):
        
        alpha = 0.00001# 1e-5 #learning rate
        self.hidden_layer_nodes1 = 20
        self.hidden_layer_nodes2 = 10

        self.state = tf.placeholder(tf.float32, [1,features] )
        #self.state_vec = tf.reshape(self.state, [1, features])
        self.Q_n = tf.placeholder(tf.float32, shape=[1, len(action_space)])
        self.W1 = tf.Variable(tf.truncated_normal([features,len(action_space)], stddev=1e-5))
        self.b1 = tf.Variable(tf.constant(0.0, shape=[len(action_space)]))
        #############################################################
        #self.z1 = tf.nn.relu(tf.add(tf.matmul(self.state,self.W1),self.b1))
        
        ##hidden layer 1:
        #self.W1 = tf.Variable(tf.truncated_normal([features,self.hidden_layer_nodes1], stddev=1e-5))
        #self.b1 = tf.Variable(tf.constant(0.0, shape=[self.hidden_layer_nodes1]))
        #self.z1 = tf.nn.relu(tf.add(tf.matmul(self.state,self.W1),self.b1))
        ##hidden layer 2:
        #self.W2 = tf.Variable(tf.truncated_normal([self.hidden_layer_nodes1,self.hidden_layer_nodes2], stddev=0.1))
        #self.b2 = tf.Variable(tf.constant(0.1, shape=[self.hidden_layer_nodes2]))
        #self.z2 = tf.nn.relu(tf.matmul(self.z1,self.W2)+self.b2)
        ##output layer:
        #self.W3 = tf.Variable(tf.truncated_normal([self.hidden_layer_nodes2,len(action_space)], stddev=0.1))
        #self.b3 = tf.Variable(tf.constant(0.0, shape=[len(action_space)]))
        #self.Q4actions = tf.matmul(self.z2,self.W3) + self.b1
        #############################################################
        self.Q4actions = tf.matmul(self.state,self.W1) + self.b1
        
        self.loss = tf.pow(self.Q4actions - self.Q_n, 2)
        self.update = tf.train.GradientDescentOptimizer(alpha).minimize(self.loss)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

       ##########################
        #self.sess.run(tf.assign(self.W1, [[-2,1],[2,-1]]))
        #self.sess.run(tf.assign(self.b1,[0,0]))
        #self.sess.run(tf.assign(self.W1, [[-1,0,1],[1,0,-1]]))
        #self.sess.run(tf.assign(self.b1,[0,0,0]))
        #l = int((len(action_space))/2)
        #self.sess.run(tf.assign(self.W1, [[x for x in range(-l,0)]+[x for x in range(0,l+1)],[x for x in range(0,l+1)]+[x for x in range(-l,0)]]))
        #self.sess.run(tf.assign(self.b1,[0 for _ in range(0,2*l +1)]))
      #########################
        return

    def get_Q(self,s):
        return self.sess.run(self.Q4actions, feed_dict={self.state: [s]})
    def update_sess(self,s,Q_corrected):
        self.sess.run(self.update, feed_dict={self.state:[s], self.Q_n: Q_corrected})
        return

    def save_model(self,*args):
         if len(args) > 0:
             file_name = args[0]
         else:
            file_name = "C:\MachineLearning\MLdriverPython\MLdriverPython\model1.ckpt" #/media/windows-share/MLdriverPython/MLdriverPython/
         saver = tf.train.Saver()
         save_path = saver.save(self.sess, file_name)
         print("Model saved in file: %s" % save_path)

    def restore(self,*args):
        if args:
            file_name = args[0]
        else:
            file_name = "C:\MachineLearning\MLdriverPython\MLdriverPython\model1.ckpt"#/media/windows-share/MLdriverPython/MLdriverPython/
        # Restore variables from disk.
        saver = tf.train.Saver()
        saver.restore(self.sess, file_name)
        print("Model restored.")

    def get_par(self):
        return self.sess.run(self.W1),self.sess.run(self.b1), 








#W=tf.Variable(tf.zeros([features,1]))
#b=tf.Variable(tf.zeros([1]))
#y=tf.matmul(x,W)+b

#loss=tf.reduce_mean(tf.pow(y-y_,2))
#update=tf.train.GradientDescentOptimizer(0.001).minimize(loss)
#m=tf.Variable(0.)

##msum = tf.summary.scalar('m', m)
#losssum = tf.summary.scalar('loss', loss)
#merged = tf.summary.merge_all()



#data_x= np.asarray([[x[0],x[1],x[3],x[4],x[5],x[6],x[7],x[8]] for x in train_data_z])

#data_y= np.asarray([[x[2]] for x in train_data_z])

#sess=tf.Session()
#file_writer = tf.summary.FileWriter('./my_graph', sess.graph)
#sess.run(tf.global_variables_initializer())
#training_loss=0
#for i in range(0,10000):
#    #curr_sammary = sess.run(losssum)
#    #sw.add_summary(curr_sammary, i)
#    #sess.run(update,feed_dict={x:data_x,y_:data_y})
#    [_,curr_sammary]=sess.run([update,merged],feed_dict={x:data_x,y_:data_y})
#    file_writer.add_summary(curr_sammary,i)
#    training_loss = loss.eval(session=sess,feed_dict={x:data_x,y_:data_y})
#    if i%100 == 0:
#        print('iteration:',i,' W:', sess.run(W), ' b:', sess.run(b), ' loss:',training_loss)

#file_writer.close()
##print(sess.run(m))

## Graphic display
##train_X = np.asarray([3.3,4.4,5.5,6.71,6.93,4.168,9.779,6.182,7.59,2.167,
##                         7.042,10.791,5.313,7.997,5.654,9.27,3.1])
##train_Y = np.asarray([1.7,2.76,2.09,3.19,1.694,1.573,3.366,2.596,2.53,1.221,
##                         2.827,3.465,1.65,2.904,2.42,2.94,1.3])
##fig = plt.figure()
##ax = fig.gca(projection='3d')
##X, Y, Z = np.meshgrid([float(x[0]) for x in data_x], [float(x[1]) for x in data_x],[float(x[0]) for x in data_y])
##print([float(x[0]) for x in data_x])
##surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
##plt.show()
##plt.plot(data_x, data_y, 'ro', label='Original data')
##plt.plot(data_x, sess.run(W) * data_x + sess.run(b), label='Fitted line')
##plt.legend()
##plt.show()

