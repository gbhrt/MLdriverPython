import tensorflow as tf
from net_lib import NetLib
def policy_estimator(action_space_n,state,features_num): #define a net - input: state (and dimentions) - output: Pi - policy
    hidden_layer_nodes1 = 100
    hidden_layer_nodes2 = 50

    #linear regression:
    #theta1 = tf.Variable(tf.truncated_normal([features_num,action_space_n], stddev=1e-5))
    #theta_b1 = tf.Variable(tf.constant(0.0, shape=[action_space_n]))
    #pi = tf.nn.softmax(tf.matmul(state,theta1) + theta_b1)
    #############################################################
        
    #hidden layer 1:
    theta1 = tf.Variable(tf.truncated_normal([features_num,hidden_layer_nodes1], stddev=0.1),name = "P_th1")
    theta_b1 = tf.Variable(tf.constant(0.1, shape=[hidden_layer_nodes1]),name = "P_b1")
    theta_z1 = tf.nn.relu(tf.add(tf.matmul(state,theta1),theta_b1))
    #hidden layer 2:
    theta2 = tf.Variable(tf.truncated_normal([hidden_layer_nodes1,hidden_layer_nodes2], stddev=0.1),name = "P_th2")
    theta_b2 = tf.Variable(tf.constant(0.1, shape=[hidden_layer_nodes2]),"P_b2")
    theta_z2 = tf.nn.relu(tf.add(tf.matmul(theta_z1,theta2),theta_b2))
    #output layer:
    theta3 = tf.Variable(tf.truncated_normal([hidden_layer_nodes2,action_space_n], stddev=0.1),"P_th3")
    theta_b3 = tf.Variable(tf.constant(0.1, shape=[action_space_n]),"P_b3")
    pi = tf.add(tf.matmul(theta_z2,theta3), theta_b3)
    pi = tf.nn.softmax(pi)
    #############################################################
    return pi
def Q_estimator(action_space_n,state,features_num):#define a net - input: state (and dimentions) - output: Q - Value
    #hidden_layer_nodes1 = 100
    #hidden_layer_nodes2 = 50
    hidden_layer_nodes1 = 300
    hidden_layer_nodes2 = 200
    #linear regression:
    #W1 = tf.Variable(tf.truncated_normal([features_num,action_space_n], stddev=1e-5))
    #b1 = tf.Variable(tf.constant(0.0, shape=[action_space_n]))
    #Q = tf.matmul(state,W1) + b1
    #############################################################
    #value layers:
    #hidden layer 1:
    V_W1 = tf.Variable(tf.truncated_normal([features_num,hidden_layer_nodes1], stddev=0.1),name = "Q_W1")
    V_b1 = tf.Variable(tf.constant(0.1, shape=[hidden_layer_nodes1]),name = "Q_b1")
    V_z1 = tf.nn.relu(tf.add(tf.matmul(state,V_W1),V_b1))
    #hidden layer 2:
    V_W2 = tf.Variable(tf.truncated_normal([hidden_layer_nodes1,hidden_layer_nodes2], stddev=0.1),name = "Q_W2")
    V_b2 = tf.Variable(tf.constant(0.1, shape=[hidden_layer_nodes2]),name = "Q_b2")
    V_z2 = tf.nn.relu(tf.add(tf.matmul(V_z1,V_W2),V_b2))
    #output layer:
    V_W3 = tf.Variable(tf.truncated_normal([hidden_layer_nodes2,1], stddev=0.1),name = "Q_W3")
    V_b3 = tf.Variable(tf.constant(0.1, shape=[1]),name = "Q_b3")
    Value = tf.add(tf.matmul(V_z2,V_W3), V_b3)
    #############################################################
    #Advantage layers:
    #hidden layer 1:    
    A_W1 = tf.Variable(tf.truncated_normal([features_num,hidden_layer_nodes1], stddev=0.1),name = "Q_W1")
    A_b1 = tf.Variable(tf.constant(0.1, shape=[hidden_layer_nodes1]),name = "Q_b1")
    A_z1 = tf.nn.relu(tf.add(tf.matmul(state,A_W1),A_b1))
    #hidden layer 2:
    A_W2 = tf.Variable(tf.truncated_normal([hidden_layer_nodes1,hidden_layer_nodes2], stddev=0.1),name = "Q_W2")
    A_b2 = tf.Variable(tf.constant(0.1, shape=[hidden_layer_nodes2]),name = "Q_b2")
    A_z2 = tf.nn.relu(tf.add(tf.matmul(A_z1,A_W2),A_b2))
    #output layer:
    A_W3 = tf.Variable(tf.truncated_normal([hidden_layer_nodes2,action_space_n], stddev=0.1),name = "Q_W3")
    A_b3 = tf.Variable(tf.constant(0.1, shape=[action_space_n]),name = "Q_b3")
    Advantage = tf.add(tf.matmul(A_z2,A_W3), A_b3)
    return Value + tf.subtract(Advantage,tf.reduce_mean(Advantage,axis=1,keepdims=True))

class DQN_network(NetLib):
    def __init__(self,features_num,action_space_n,alpha_actor = None,alpha_critic = None, tau = 1.0):
        self.state = tf.placeholder(tf.float32, [None,features_num] )
        self.action = tf.placeholder( dtype=tf.int32, shape=[None] )
        self.actions_one_hot = tf.one_hot(self.action,action_space_n,dtype=tf.float32)#one hot
        self.input_targetQa = tf.placeholder(tf.float32, shape=[None])

        self.Q = Q_estimator(action_space_n, self.state,features_num)
        self.targetQ = Q_estimator(action_space_n, self.state,features_num)
        network_params = tf.trainable_variables()
        params_num = len(network_params)
        Q_params = network_params[:params_num//2]
        target_Q_params = network_params[params_num//2:]
        if alpha_actor !=None and alpha_critic != None:#for training the network
            self.Qa = tf.reduce_sum(tf.multiply(self.Q, self.actions_one_hot),axis=1) #Q on action a at the feeded state
            
            self.Q_loss = tf.squared_difference(self.Qa,self.input_targetQa)
            self.update_Q = tf.train.AdamOptimizer(alpha_critic).minimize(self.Q_loss)
            self.update_var_vec = self.update_target_init(tau,Q_params,target_Q_params)
        
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        return
    def get_Q(self,s):#get one Q at a specific state
        Q = self.sess.run(self.Q, feed_dict={self.state: s})
        return Q
    def get_targetQ(self,s):#get one pi at a specific state
        Q = self.sess.run(self.targetQ, feed_dict={self.state: s})
        return Q

    def Update_Q(self,s,a,input_targetQa):
        self.sess.run(self.update_Q, feed_dict={self.state: s ,self.action: a ,self.input_targetQa: input_targetQa})# 
        return 
    def get_Q_loss(self,s,a,input_targetQa):
        return self.sess.run(self.Q_loss ,feed_dict={self.state: [s], self.input_targetQa: [input_targetQa],self.action:a})

    def update_target(self):
        for var in self.update_var_vec:
            self.sess.run(var)