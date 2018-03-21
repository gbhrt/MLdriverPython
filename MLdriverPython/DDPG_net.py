import tensorflow as tf
from net_lib import NetLib

def continues_actor(action_n,action_limit,state,state_n): #define a net - input: state (and dimentions) - output: a continues action
    hidden_layer_nodes1 = 100
    hidden_layer_nodes2 = 50

    #hidden layer 1:
    theta1 = tf.Variable(tf.truncated_normal([state_n,hidden_layer_nodes1], stddev=0.1),name = "P_th1")
    theta_b1 = tf.Variable(tf.constant(0.1, shape=[hidden_layer_nodes1]),name = "P_b1")
    theta_z1 = tf.nn.relu(tf.add(tf.matmul(state,theta1),theta_b1))
    #hidden layer 2:
    theta2 = tf.Variable(tf.truncated_normal([hidden_layer_nodes1,hidden_layer_nodes2], stddev=0.1),name = "P_th2")
    theta_b2 = tf.Variable(tf.constant(0.1, shape=[hidden_layer_nodes2]),"P_b2")
    theta_z2 = tf.nn.relu(tf.add(tf.matmul(theta_z1,theta2),theta_b2))
    #output layer:
    theta3 = tf.Variable(tf.truncated_normal([hidden_layer_nodes2,action_n], stddev=0.1),"P_th3")
    theta_b3 = tf.Variable(tf.constant(0.1, shape=[action_n]),"P_b3")
    action = tf.multiply(tf.nn.tanh(tf.add(tf.matmul(theta_z2,theta3), theta_b3)),action_limit)
    return action

def continues_critic(action,action_n,state,state_n):#define a net - input: state (and dimentions) - output: Q - Value
    hidden_layer_nodes1 = 100
    hidden_layer_nodes2 = 50

    #linear regression:
    #W1 = tf.Variable(tf.truncated_normal([features_num,action_space_n], stddev=1e-5))
    #b1 = tf.Variable(tf.constant(0.0, shape=[action_space_n]))
    #Q = tf.matmul(state,W1) + b1
    #############################################################
    #hidden layer 1 - input state
    W1_1 = tf.Variable(tf.truncated_normal([state_n,hidden_layer_nodes1], stddev=0.1),name = "Q_W1")
    b1_1 = tf.Variable(tf.constant(0.1, shape=[hidden_layer_nodes1]),name = "Q_b1")
    z1_1 = tf.nn.relu(tf.add(tf.matmul(state,W1_1),b1_1))

    #hidden layer 1 - input actions:
    W1_2 = tf.Variable(tf.truncated_normal([action_n,hidden_layer_nodes1], stddev=0.1),name = "Q_W1")
    b1_2 = tf.Variable(tf.constant(0.1, shape=[hidden_layer_nodes1]),name = "Q_b1")
    z1_2 = tf.nn.relu(tf.add(tf.matmul(action,V_W1),V_b1))


    #hidden layer 2:
    W2 = tf.Variable(tf.truncated_normal([hidden_layer_nodes1,hidden_layer_nodes2], stddev=0.1),name = "Q_W2")
    b2 = tf.Variable(tf.constant(0.1, shape=[hidden_layer_nodes2]),name = "Q_b2")
    z2 = tf.nn.relu(tf.add(tf.matmul(z1_1,V_W2),b2) + tf.add(tf.matmul(z1_2,V_W2),b2))#add the two first layers

    #output layer:
    W3 = tf.Variable(tf.truncated_normal([hidden_layer_nodes2,action_n], stddev=0.1),name = "Q_W3")
    b3 = tf.Variable(tf.constant(0.1, shape=[action_n]),name = "Q_b3")
    Qa = tf.add(tf.matmul(z2,W3), b3)
       
    return Qa

class DDPG_network(NetLib):
    def DDPG(self,state_n,action_space_n,action_limit,alpha_actor = None,alpha_critic = None, tau = 1.0):
        self.state = tf.placeholder(tf.float32, [None,state_n] )
        self.V_ = tf.placeholder(tf.float32,shape=[None])
        self.action_in = tf.placeholder( dtype=tf.float32, shape=[None,action_space_n] )
        self.targetQa_in = tf.placeholder(tf.float32, shape=[None])

        self.action_out = continues_actor(action_space_n,action_limit,self.state,state_n)
        self.target_action_out = continues_actor(action_space_n,action_limit,self.state,state_n)
        self.Qa = continues_critic(self.action_in,action_space_n, self.state,state_n)
        self.target_Qa = continues_critic(self.action_in,action_space_n, self.state,state_n)

        if alpha_actor !=None and alpha_critic != None:#for training the network
            #critic loss
            self.Q_loss = tf.squared_difference(self.Qa,self.targetQa_in)
            self.update_Q = tf.train.AdamOptimizer(alpha_critic).minimize(self.Q_loss)

            #actor loss:
            action_gradient = tf.gradients(action_out, self.action_in)

            self.actor_loss = tf.reduce_mean()
            self.update_actor = tf.train.AdamOptimizer(alpha_actor).apply_gradients(zip(self.actor_gradients, self.network_params))

            self.update_target_init(tau)

            self.Pia = tf.reduce_sum(tf.multiply(self.Pi, self.actions_one_hot),axis=1)#Pi on action a at the feeded state

            self.policy_loss = - tf.log(self.Pia+1e-8)*self.V_
            self.update_policy = tf.train.AdamOptimizer(alpha_actor).minimize(self.policy_loss)

        return

    def get_Qa(self,s,a):#get Q(s,a) at a specific state
        Q = self.sess.run(self.Qa, feed_dict={self.state: s,self.action_in: a})
        return Q
    def get_targetQa(self,s,a):#get Q(s,a) at a specific state
        Q = self.sess.run(self.target_Qa, feed_dict={self.state: s,self.action_in: a})
        return Q
    def Update_critic(self,s,a,targetQa_in):
        self.sess.run(self.update_Q, feed_dict={self.state: s ,self.action_in: a ,self.targetQa_in: targetQa_in})# 
        return 