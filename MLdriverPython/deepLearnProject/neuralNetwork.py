import tensorflow as tf
#from mpl_toolkits.mplot3d import Axes3D
#from matplotlib import cm
#import matplotlib.pyplot as plt
import numpy as np
import random
import math
import os, sys
from library import *
import _thread

class neuralNetwork():
    def __init__(self):
        self.features_num = 2
        self.hidden_layer_nodes1 = 20
        self.hidden_layer_nodes2 = 10

        self.alpha = 0.001
        self.x=tf.placeholder(tf.float32, [None,self.features_num])
        self.y_=tf.placeholder(tf.float32,[None,1])


        self.W1 = tf.Variable(tf.truncated_normal([self.features_num,self.hidden_layer_nodes1], stddev=0.1))
        self.b1 = tf.Variable(tf.constant(0.1, shape=[self.hidden_layer_nodes1]))
        self.z1 = tf.tanh(tf.add(tf.matmul(self.x,self.W1),self.b1))

        self.W2 = tf.Variable(tf.truncated_normal([self.hidden_layer_nodes1,self.hidden_layer_nodes2], stddev=0.1))
        self.b2 = tf.Variable(tf.constant(0.1, shape=[self.hidden_layer_nodes2]))
        self.z2 = tf.tanh(tf.matmul(self.z1,self.W2)+self.b2)
        #y=tf.matmul(z1,W2)+b2

        self.W3 = tf.Variable(tf.truncated_normal([self.hidden_layer_nodes2,1], stddev=0.1))
        self.b3 = tf.Variable(0.)
        #z2 = tf.tanh(tf.matmul(x,W2)+b2)
        self.y = tf.matmul(self.z2,self.W3)+self.b3

        self.loss=tf.reduce_mean(tf.pow(self.y-self.y_,2))
        self.update=tf.train.GradientDescentOptimizer(self.alpha).minimize(self.loss)
        self.losssum = tf.summary.scalar('loss', self.loss)
        self.merged = tf.summary.merge_all()

        self.sess=tf.Session()

    def read_data(self,file_name):#, x,y,steer_ang
        with open(file_name, 'r') as f:#append data to the file
            data = f.readlines()
            data = [x.strip().split() for x in data]
            results = []
            for x in data:
                results.append(list(map(float, x)))

        return results

    def input_thread(self,stop):
        input()
        stop.append(True)
        return
    def wait_for(self,stop):
        _thread.start_new_thread(input_thread, (stop,))
        return



    def train_network(self):
        stop = []
        command =[]
        wait_for(stop,command)#wait for "enter" in annother thread - then stop = true

        #data1.txt  # start_steering, end_pos_x end_pos_y, end_angle, end_steering - new data
        data = read_data("train_data2.txt") # x y steering - old data

        #split data to train and test data
        random.shuffle(data)
        train_data = data[:int(0.75*len(data))]
        test_data = data[int(0.25*len(data)):]

        #features_num = self.features_num #old data
        data_x= np.asarray([[x[0],x[1]] for x in train_data])
        data_y= np.asarray([[x[2]] for x in train_data])

        file_writer = tf.summary.FileWriter('./my_graph', self.sess.graph)
        self.sess.run(tf.global_variables_initializer())
        training_loss=0
        for i in range(100000):
            if stop:
                break
            #curr_sammary = self.sess.run(losssum)
            #sw.add_summary(curr_sammary, i)
            #self.sess.run(update,feed_dict={x:data_x,y_:data_y})
            [_,curr_sammary]=self.sess.run([self.update,self.merged],feed_dict={self.x:data_x,self.y_:data_y})
            file_writer.add_summary(curr_sammary,i)
            training_loss = self.loss.eval(session=self.sess,feed_dict={self.x:data_x,self.y_:data_y})
            if i%100 == 0:
                print(i,' loss:',training_loss)
            #print('iteration:',i,' W1:', self.sess.run(W1),' W2:', self.sess.run(W2), ' b1:', self.sess.run(b1),' b2:', self.sess.run(b2), ' loss:',training_loss)

        file_writer.close()
        print('iteration:',i,' W1:', self.sess.run(self.W1),' W2:', self.sess.run(self.W2), ' b1:', self.sess.run(self.b1),' b2:', self.sess.run(self.b2), ' loss:',training_loss)
        #print(self.sess.run(m))

        # Graphic display
        #train_X = np.asarray([3.3,4.4,5.5,6.71,6.93,4.168,9.779,6.182,7.59,2.167,
        #                         7.042,10.791,5.313,7.997,5.654,9.27,3.1])
        #train_Y = np.asarray([1.7,2.76,2.09,3.19,1.694,1.573,3.366,2.596,2.53,1.221,
        #                         2.827,3.465,1.65,2.904,2.42,2.94,1.3])
        #fig = plt.figure()
        #ax = fig.gca(projection='3d')
        #X, Y, Z = np.meshgrid([float(x[0]) for x in data_x], [float(x[1]) for x in data_x],[float(x[0]) for x in data_y])
        #print([float(x[0]) for x in data_x])
        #surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
        #plt.show()
        #plt.plot(data_x, data_y, 'ro', label='Original data')
        #plt.plot(data_x, self.sess.run(W) * data_x + self.sess.run(b), label='Fitted line')
        #plt.legend()
        #plt.show()

        # Testing example, as requested (Issue #2)


        test_X= np.asarray([[x[0],x[1]] for x in test_data]) # old data
        test_Y= np.asarray([[x[2]] for x in test_data])

        print("Testing... (Mean square loss Comparison)")
        testing_cost = self.sess.run(tf.reduce_mean(tf.pow(self.y-self.y_,2)), feed_dict={self.x: test_X, self.y_: test_Y})  # same function as cost above
        print("Training loss=", training_loss)
        print("Testing cost=", testing_cost)
        print("mean square loss difference:", training_loss - testing_cost)



        #file_name = "/media/windows-share/MLdriverPython/MLdriverPython/model1.ckpt" #os.path.join(os.getcwd(), 'model.ckpt') #/media/windows-share/MLdriverPython/MLdriverPython/
        #saver = tf.train.Saver()
        #save_path = saver.save(self.sess, file_name)
        #print("Model saved in file: %s" % save_path)

        
        #plt.plot(test_X, test_Y, 'bo', label='Testing data')
        #plt.plot(data_x, self.sess.run(W) * data_x + self.sess.run(b), label='Fitted line')
        #plt.legend()
        #plt.show()

    def restore_session(self):
        file_name = "/media/windows-share/MLdriverPython/MLdriverPython/model1.ckpt" #os.path.join(os.getcwd(), 'model.ckpt')#"model.ckpt" #/media/windows-share/MLdriverPython/MLdriverPython/
        # Restore variables from disk.
        saver = tf.train.Saver()
        saver.restore(self.sess, file_name)
        print("Model restored.")

    def predict(self, point): #point = [x,y] 
        t = self.sess.run(self.y, feed_dict = {self.x:[point]})
        
        return float(t[0])

if __name__ == "__main__": 
    nn = neuralNetwork()
    #restore = False
    #if len(sys.argv)>1:
    #    try:
    #        restore = bool(sys.argv[1])
    #    except:
    #        pass

    #if restore:
    #    nn.restore_session()
    
    nn.train_network()
    #data = read_data("train_data2.txt") # x y steering - old data

    ##split data to train and test data
    #random.shuffle(data)
    #test_data = data[int(0.25*len(data)):]
    #test_X= np.asarray([[x[0],x[1]] for x in test_data]) # old data
    #test_Y= np.asarray([[x[2]] for x in test_data])
    #with open("prediction_error.txt",'w') as f:
    #    for i in range (len(test_X)):
    #        k = nn.predict([test_X[i][0],test_X[i][1]])
    #        f.write("%s\t%s\t%s\t%s\t%s\n" % (test_X[i][0],test_X[i][1],k,test_Y[i][0],k - test_Y[i][0]))

    #k = nn.predict([-0.5, 5.233044858679446])
    #print(k)
