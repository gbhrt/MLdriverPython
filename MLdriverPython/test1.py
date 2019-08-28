# To plot pretty figures
#%matplotlib inline
#import matplotlib
#import matplotlib.pyplot as plt
#import pandas as pd
import numpy as np
import tensorflow as tf
import threading



#print(tf.__version__)
#input()
#n_inputs = 4
#n_neurons = 6
#n_timesteps = 2

#def create_ts(start = '2001', n = 201, freq = 'M'):
#    rng = pd.date_range(start=start, periods=n, freq=freq)
#    ts = pd.Series(np.random.uniform(-18, 18, size=len(rng)), rng).cumsum()
#    return ts
#ts= create_ts(start = '2001', n = 192, freq = 'M')
#ts.tail(5)

#ts = create_ts(start = '2001', n = 222)





## Left
#plt.figure(figsize=(11,4))
#plt.subplot(121)
#plt.plot(ts.index, ts)
#plt.plot(ts.index[90:100], ts[90:100], "b-", linewidth=3, label="A training instance")
#plt.title("A time series (generated)", fontsize=14)

## Right
#plt.subplot(122)
#plt.title("A training instance", fontsize=14)
#plt.plot(ts.index[90:100], ts[90:100], "b-", markersize=8, label="instance")
#plt.plot(ts.index[91:101], ts[91:101], "bo", markersize=10, label="target", markerfacecolor='red')
#plt.legend(loc="upper left")
#plt.xlabel("Time")

#plt.show()



#series = np.array(ts)
#n_windows = 20   
#n_input =  1
#n_output = 1
#size_train = 201

### Split data
#train = series[:size_train]
#test = series[size_train:]
#print(train.shape, test.shape)

###x_data = train[:size_train-1] #Select all the training instance minus one day
###X_batches = x_data.reshape(-1, windows, input) #create the right shape for the batch e.g (10, 20, 1)
#def create_batches(df, windows, input, output):
#    ## Create X         
#        x_data = train[:size_train-1] # Select the data
#        X_batches = x_data.reshape(-1, windows, input)  # Reshape the data 
#    ## Create y
#        y_data = train[n_output:size_train]
#        y_batches = y_data.reshape(-1, windows, output)
#        return X_batches, y_batches

#X_batches, y_batches = create_batches(df = train,
#                                      windows = n_windows,
#                                      input = n_input,
#                                      output = n_output)
#X_test, y_test = create_batches(df = test, windows = 20,input = 1, output = 1)

#tf.reset_default_graph()
#r_neuron = 120 

#X = tf.placeholder(tf.float32, [None, n_windows, n_input])   
#y = tf.placeholder(tf.float32, [None, n_windows, n_output])

#basic_cell = tf.contrib.rnn.BasicRNNCell(num_units=r_neuron, activation=tf.nn.relu)   
#rnn_output, states = tf.nn.dynamic_rnn(basic_cell, X, dtype=tf.float32) 

#stacked_rnn_output = tf.reshape(rnn_output, [-1, r_neuron])          
#stacked_outputs = tf.layers.dense(stacked_rnn_output, n_output)       
#outputs = tf.reshape(stacked_outputs, [-1, n_windows, n_output])  

#learning_rate = 0.001  
 
#loss = tf.reduce_sum(tf.square(outputs - y))    
#optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)         
#training_op = optimizer.minimize(loss)                                          

#init = tf.global_variables_initializer() 

#iteration =4500 

#with tf.Session() as sess:
#    init.run()
#    for iters in range(iteration):
#        sess.run(training_op, feed_dict={X: X_batches, y: y_batches})
#        if iters % 150 == 0:
#            mse = loss.eval(feed_dict={X: X_batches, y: y_batches})
#            print(iters, "\tMSE:", mse)
    
#    y_pred = sess.run(outputs, feed_dict={X: X_test})
    
#plt.title("Forecast vs Actual", fontsize=14)
#plt.plot(pd.Series(np.ravel(y_test)), "bo", markersize=8, label="Actual", color='green')
#plt.plot(pd.Series(np.ravel(y_pred)), "r.", markersize=8, label="Forecast", color='red')
#plt.legend(loc="lower left")
#plt.xlabel("Time")

#plt.show()  



model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(20, activation=tf.keras.activations.relu,input_shape = (2,) ),
tf.keras.layers.Dense(20, activation=tf.keras.activations.relu),
tf.keras.layers.Dense(20, activation=tf.keras.activations.relu),
tf.keras.layers.Dense(1)
])

model.compile(optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.mean_squared_error,
        metrics=['accuracy'])

graph = tf.get_default_graph()

model.summary()
print(model.evaluate(np.array([[1,2],[2,4]]),np.array([1,2])))
model.train_on_batch(np.array([[1,2],[2,4]]),np.array([1,2]))
print(model.predict(np.array([[1,2],[2,4]]),batch_size = 2))#


def train():
    for i in range(100):
        with graph.as_default():
            model.train_on_batch(np.array([[1,2],[2,4]]),np.array([1,2]))
            print("train model")


t = threading.Thread(target=train)
t.start()


