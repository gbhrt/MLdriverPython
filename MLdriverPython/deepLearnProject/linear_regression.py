import tensorflow as tf
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
import random
import math
from library import *

file_name = "train_data2.txt"
data = read_data(file_name) # x,y,steer_ang

random.shuffle(data)

train_data = data[:int(0.75*len(data))]
test_data = data[int(0.25*len(data)):]
train_data_z = []
test_data_z = []
#i=0
#def fit_data(feature):
#    max_ = max(feature)
#    min_ = min(feature)
#    divider = max_ - min_
for x in train_data:
    #if x[0]**2+x[1]**2 > 0:
    #    z = math.atan(x[0]/(x[0]**2+x[1]**2))
    #else:
    #    z=0
    x.append(x[0]**2/25)
    x.append(x[1]**2/25)
    x.append(x[0]**3/10000)
    x.append(x[1]**3/10000)
    x.append(x[0]**4/100000)
    x.append(x[1]**4/100000)
    #x.append(0.)
    #x.append(0.)
    #print("x: ", x,"\n")
    
    train_data_z.append(x)
    #z= x[0]**2
    #train_data_z[-1].append(z)
    #z= x[1]**2
    #train_data_z[-1].append(z)
#print(train_data_z)
#input("press enter")
for x in test_data:
    #if x[0]**2+x[1]**2 > 0:
    #    z = math.atan(x[0]/(x[0]**2+x[1]**2))
    #else:
    #    z=0
    #test_data_z.append(x)
    #test_data_z[-1].append(z)
    x.append(x[0]**2/25)
    x.append(x[1]**2/25)
    x.append(x[0]**3/10000)
    x.append(x[1]**3/10000)
    x.append(x[0]**4/100000)
    x.append(x[1]**4/100000)
    #x.append(0.)
    #x.append(0.)
    test_data_z.append(x)



features = 8
x=tf.placeholder(tf.float32, [None,features])
y_=tf.placeholder(tf.float32,[None,1])

W=tf.Variable(tf.zeros([features,1]))
b=tf.Variable(tf.zeros([1]))
y=tf.matmul(x,W)+b

loss=tf.reduce_mean(tf.pow(y-y_,2))
update=tf.train.GradientDescentOptimizer(0.001).minimize(loss)
m=tf.Variable(0.)

#msum = tf.summary.scalar('m', m)
losssum = tf.summary.scalar('loss', loss)
merged = tf.summary.merge_all()



data_x= np.asarray([[x[0],x[1],x[3],x[4],x[5],x[6],x[7],x[8]] for x in train_data_z])
#data_x= np.asarray([[x[0],x[1]] for x in train_data_z])
data_y= np.asarray([[x[2]] for x in train_data_z])

sess=tf.Session()
file_writer = tf.summary.FileWriter('./my_graph', sess.graph)
sess.run(tf.global_variables_initializer())
training_loss=0
for i in range(0,10000):
    #curr_sammary = sess.run(losssum)
    #sw.add_summary(curr_sammary, i)
    #sess.run(update,feed_dict={x:data_x,y_:data_y})
    [_,curr_sammary]=sess.run([update,merged],feed_dict={x:data_x,y_:data_y})
    file_writer.add_summary(curr_sammary,i)
    training_loss = loss.eval(session=sess,feed_dict={x:data_x,y_:data_y})
    if i%100 == 0:
        print('iteration:',i,' W:', sess.run(W), ' b:', sess.run(b), ' loss:',training_loss)

file_writer.close()
#print(sess.run(m))

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
#plt.plot(data_x, sess.run(W) * data_x + sess.run(b), label='Fitted line')
#plt.legend()
#plt.show()

# Testing example, as requested (Issue #2)
#test_X= np.asarray([[x[0],x[1],x[3],x[4]] for x in test_data_z])
test_X= np.asarray([[x[0],x[1],x[3],x[4],x[5],x[6],x[7],x[8]] for x in test_data_z])
test_Y= np.asarray([[x[2]] for x in test_data_z])

print("Testing... (Mean square loss Comparison)")
testing_cost = sess.run(tf.reduce_mean(tf.pow(y-y_,2)), feed_dict={x: test_X, y_: test_Y})  # same function as cost above
print("Training loss=", training_loss)
print("Testing cost=", testing_cost)
print("Absolute mean square loss difference:", abs(training_loss - testing_cost))


#plt.plot(test_X, test_Y, 'bo', label='Testing data')
#plt.plot(data_x, sess.run(W) * data_x + sess.run(b), label='Fitted line')
#plt.legend()
#plt.show()

