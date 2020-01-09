import tensorflow as tf
import tflearn
from net_lib import NetLib
#tensorboard --logdir=C:\Users\gavri\Desktop\MLdriverPython



class DDPG_network(NetLib):
    def __init__(self,state_n,action_space_n,\
        alpha_actor = None,alpha_critic = None,alpha_analytic_actor = None,alpha_analytic_critic = None,  tau = 1.0,seed = None,conv_flag = True,feature_data_n = 1):
       

        tf.reset_default_graph()     
        self.graph = tf.get_default_graph()# tf.get_default_graph()
        if seed != None:
            tf.set_random_seed(seed)
        self.state = tf.placeholder(tf.float32, [None,state_n] )
        self.action_in = tf.placeholder( dtype=tf.float32, shape=[None,action_space_n] )
        self.targetQa_in = tf.placeholder(tf.float32, shape=[None,1])
        self.action_out = self.continues_actor(action_space_n,self.state,state_n,feature_data_n = feature_data_n, conv_flag = conv_flag)
        network_params = tf.trainable_variables()
        self.target_action_out = self.continues_actor(action_space_n,self.state,state_n,feature_data_n = feature_data_n,conv_flag = conv_flag)
        network_params = tf.trainable_variables()
        params_num = len(network_params)
        self.actor_params = network_params[:params_num//2]
        self.target_actor_params = network_params[params_num//2:]
        
        self.Qa = self.continues_critic(self.action_in,action_space_n, self.state,state_n,feature_data_n = feature_data_n,conv_flag = conv_flag)
        self.target_Qa = self.continues_critic(self.action_in,action_space_n, self.state,state_n,feature_data_n = feature_data_n,conv_flag = conv_flag)
        network_params = tf.trainable_variables()[params_num:]
        params_num = len(network_params)
        self.critic_params = network_params[:params_num//2]
        self.target_critic_params = network_params[params_num//2:]
        

        if alpha_actor !=None and alpha_critic != None:#for training the network
            #critic loss
            self.Q_loss = tf.reduce_mean( tf.squared_difference(self.Qa,self.targetQa_in))#
            self.update_critic = tf.train.AdamOptimizer(alpha_critic).minimize(self.Q_loss)

            #actor loss:
            self.Q_grads = tf.gradients(self.Qa, self.action_in)#gradient of Q(s,a) w.r.t a
            # Combine the gradients here
            #miu_grads = tf.gradients(self.action_out, actor_params)
            self.Q_grads_neg = list(map(lambda x: -x, self.Q_grads))
            #actor_gradients = zip(Q_grads,miu_grads)
            self.actor_gradients = tf.gradients(self.action_out, self.actor_params, self.Q_grads_neg)

            batch_size = tf.cast((tf.size(self.state)/state_n),tf.float32)
   
            #batch_size = 10
            #self.norm_actor_gradients = list(map(lambda x: tf.div(x,batch_size), self.actor_gradients))#normalize
            #self.actor_gradients.shape()
            self.norm_actor_gradients = []
            for x in self.actor_gradients:
                self.norm_actor_gradients.append(tf.div(x,batch_size))

            #self.update_actor = tf.train.AdamOptimizer(alpha_actor).apply_gradients(zip(self.actor_gradients, self.actor_params))
            self.update_actor = tf.train.AdamOptimizer(alpha_actor).apply_gradients(zip(self.norm_actor_gradients, self.actor_params))

            self.update_actor_target_vec = self.update_target_init(tau,self.actor_params,self.target_actor_params)#update targrt networks slowly (tau)
            self.update_critic_target_vec = self.update_target_init(tau,self.critic_params,self.target_critic_params)

            #############################################################
            #for analytic initialize:
            #self.analytic_action = tf.placeholder( dtype=tf.float32, shape=[None,action_space_n] )
            #self.analytic_actor_loss = tf.reduce_mean( tf.squared_difference(self.action_out,self.analytic_action))#
            #self.update_analytic_actor = tf.train.AdamOptimizer(alpha_analytic_actor).minimize(self.analytic_actor_loss)

            #self.copy_actor_target_vec = self.copy_target_init(tau,self.actor_params,self.target_actor_params)#update targrt networks immediately
            #self.copy_critic_target_vec = self.copy_target_init(tau,self.critic_params,self.target_critic_params)
            #############################################################
            #self.Pia = tf.reduce_sum(tf.multiply(self.Pi, self.actions_one_hot),axis=1)#Pi on action a at the feeded state

            #self.policy_loss = - tf.log(self.Pia+1e-8)*self.V_
            #self.update_policy = tf.train.AdamOptimizer(alpha_actor).minimize(self.policy_loss)
            #self.losssum = tf.summary.scalar('loss', self.Q_loss)
            self.init_summaries()
            #tf.summary.histogram('grads',self.Q_grads)
            #self.merged = tf.summary.merge_all()

            self.sess = tf.Session()
            self.sess.run(tf.global_variables_initializer())
            #model = tflearn.DNN(self.action_out, tensorboard_verbose=3)
            #file_writer = tf.summary.FileWriter(r'C:\Users\gavri\Desktop\MLdriverPython\MLdriverPython\MLdriverPython\my_graph', self.sess.graph)
            #print("size: ",self.sess.run(batch_size, feed_dict = {self.state: [[0 for _ in range(31)] for _ in range(40)]}))
            print("Network ready")
        return
    def continues_actor(self,action_n,state,state_n,feature_data_n = 1,conv_flag = True): #define a net - input: state (and dimentions) - output: a continues action ,
        hidden_layer_nodes1 = 400#400
        hidden_layer_nodes2 = 300#300
        #hidden_layer_nodes1 = 40#400
        #hidden_layer_nodes2 = 30#300

        #hidden_layer_nodes3 = 200

        ##hidden layer 1:
        #theta1 = tf.Variable(tf.truncated_normal([state_n,hidden_layer_nodes1], stddev=0.02),name = "P_th1")
        #theta_b1 = tf.Variable(tf.constant(0.0, shape=[hidden_layer_nodes1]),name = "P_b1")
        #theta_z1 = tf.nn.relu(tf.add(tf.matmul(state,theta1),theta_b1))
        ##hidden layer 2:
        #theta2 = tf.Variable(tf.truncated_normal([hidden_layer_nodes1,hidden_layer_nodes2], stddev=0.02),name = "P_th2")
        #theta_b2 = tf.Variable(tf.constant(0.0, shape=[hidden_layer_nodes2]),"P_b2")
        #theta_z2 = tf.nn.relu(tf.add(tf.matmul(theta_z1,theta2),theta_b2))
        ##output layer:
        #theta3 = tf.Variable(tf.random_uniform([hidden_layer_nodes2,action_n], minval = -0.003, maxval = 0.003),"P_th3") 
        #theta_b3 = tf.Variable(tf.constant(0.0, shape=[action_n]),"P_b3") #tf.random_uniform([action_n], minval = -0.003, maxval = 0.003)
        #action = tf.multiply(tf.nn.tanh(tf.add(tf.matmul(theta_z2,theta3), theta_b3)),action_limit)

        #input = tflearn.input_data(shape=[None, state_n])


        #if conv_flag:#if add convolution layers - split state to data and path (or picture)
        #    #if the first item in each state is the analytical action:
        #    s = state[...,feature_data_n:]
        #    data_s = state[...,0:feature_data_n]
        #    s_n = state_n - feature_data_n#data items removed


        #    s = tf.reshape(s, [-1, s_n, 1])#
        #    conv = tflearn.layers.conv.conv_1d(s,16,6,activation = 'relu')
        #    #net = tf.reshape(net, [-1,net.W.get_shape().as_list()[0]])
        #    #net = net[...,0:-2]
        #    #batch_size.shape()
        #    # Flatten the data to a 1-D vector for the fully connected layer
        #    conv = tf.contrib.layers.flatten(conv)
        #    hidden_data_layer_nodes = 50
        #    fc_data = tflearn.fully_connected(data_s, hidden_data_layer_nodes,regularizer='L2', weight_decay=0.01)#
        #    fc1_1 = tflearn.fully_connected(conv, hidden_layer_nodes1,regularizer='L2', weight_decay=0.01)#from state
        #    fc1_2 = tflearn.fully_connected(fc_data, hidden_layer_nodes1,regularizer='L2', weight_decay=0.01)#from analytic action
        #    fc1 = tflearn.activation(tf.matmul(conv, fc1_1.W) + tf.matmul(fc_data, fc1_2.W) + fc1_1.b + fc1_2.b, activation='relu')
        #    #state[...,1:].shape()

        if conv_flag:# split state to data and path (or picture)
            #if the first item in each state is the analytical action:
            path_s = state[...,feature_data_n:]
            data_s = state[...,0:feature_data_n]
            s_n = state_n - feature_data_n#data items removed


            path_s = tf.reshape(path_s, [-1, s_n, 1])#
            hidden_layer_nodes_path1 = 20
            hidden_layer_nodes_path2 = 5
            path_fc1 = tflearn.fully_connected(path_s, hidden_layer_nodes_path1,regularizer='L2', weight_decay=0.01)

            path_fc2 = tflearn.fully_connected(path_fc1, hidden_layer_nodes_path1,regularizer='L2', weight_decay=0.01)
            #conv = tflearn.layers.conv.conv_1d(s,16,6,activation = 'relu')

            #conv = tf.contrib.layers.flatten(conv)
            #hidden_data_layer_nodes = 50
            #fc_data = tflearn.fully_connected(data_s, hidden_data_layer_nodes,regularizer='L2', weight_decay=0.01)#
            fc1_1 = tflearn.fully_connected(path_fc2, hidden_layer_nodes1,regularizer='L2', weight_decay=0.01)#from state
            fc1_2 = tflearn.fully_connected(data_s, hidden_layer_nodes1,regularizer='L2', weight_decay=0.01)#from analytic action
            fc1 = tflearn.activation(tf.matmul(path_fc2, fc1_1.W) + tf.matmul(data_s, fc1_2.W) + fc1_1.b + fc1_2.b, activation='relu')
            #state[...,1:].shape()
        else:
            fc1 = tflearn.fully_connected(state, hidden_layer_nodes1,regularizer='L2', weight_decay=0.01)#from state
            #fc1 = tflearn.layers.normalization.batch_normalization(fc1)
            fc1 = tflearn.activations.relu(fc1)

        fc2 = tflearn.fully_connected(fc1, hidden_layer_nodes2,regularizer='L2', weight_decay=0.01)
        #net = tflearn.layers.normalization.batch_normalization(net)
        fc2 = tflearn.activations.relu(fc2)
        
        #fc3 = tflearn.fully_connected(fc2, hidden_layer_nodes3,regularizer='L2', weight_decay=0.01)
        #fc3 = tflearn.activations.relu(fc3)

        init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)# Final layer weights are init to Uniform[-3e-3, 3e-3]
        action = tflearn.fully_connected(fc2, action_n, activation='tanh', weights_init=init,bias_init = init,regularizer='L2', weight_decay=0.01)
        # Scale output to -action_bound to action_bound
        #action = tf.multiply(action, action_limit)

        return action
    def continues_critic(self,action,action_n,state,state_n,feature_data_n = 1,conv_flag = True):#define a net - input: state (and dimentions) - output: Q - Value
        hidden_layer_nodes1 = 400#400
        hidden_layer_nodes2 = 300#300

        #if conv_flag:#if add convolution layers - split state to data and path (or picture)
        #    #if the first item in each state is the analytical action:
        #    s = state[...,feature_data_n:]
        #    data_s = state[...,0:feature_data_n]
        #    s_n = state_n - feature_data_n#data items removed


        #    s = tf.reshape(s, [-1, s_n, 1])#
        #    conv = tflearn.layers.conv.conv_1d(s,16,6,activation = 'relu')
        #    #net = tf.reshape(net, [-1,net.W.get_shape().as_list()[0]])
        #    #net = net[...,0:-2]
        #    #batch_size.shape()
        #    # Flatten the data to a 1-D vector for the fully connected layer
        #    conv = tf.contrib.layers.flatten(conv)

        #    hidden_data_layer_nodes = 50
        #    fc_data = tflearn.fully_connected(data_s, hidden_data_layer_nodes,regularizer='L2', weight_decay=0.01)#

        #    fc1_1 = tflearn.fully_connected(conv, hidden_layer_nodes1,regularizer='L2', weight_decay=0.01)#from state
        #    fc1_2 = tflearn.fully_connected(fc_data, hidden_layer_nodes1,regularizer='L2', weight_decay=0.01)#from analytic action
        #    fc1 = tflearn.activation(tf.matmul(conv, fc1_1.W) + tf.matmul(fc_data, fc1_2.W) + fc1_1.b + fc1_2.b, activation='relu')
        if conv_flag:#if add convolution layers - split state to data and path (or picture)
            #if the first item in each state is the analytical action:
            path_s = state[...,feature_data_n:]
            data_s = state[...,0:feature_data_n]
            s_n = state_n - feature_data_n#data items removed


            path_s = tf.reshape(path_s, [-1, s_n, 1])#
            hidden_layer_nodes_path1 = 20
            hidden_layer_nodes_path2 = 5
            path_fc1 = tflearn.fully_connected(path_s, hidden_layer_nodes_path1,regularizer='L2', weight_decay=0.01)

            path_fc2 = tflearn.fully_connected(path_fc1, hidden_layer_nodes_path1,regularizer='L2', weight_decay=0.01)
            #conv = tflearn.layers.conv.conv_1d(s,16,6,activation = 'relu')

            #conv = tf.contrib.layers.flatten(conv)
            #hidden_data_layer_nodes = 50
            #fc_data = tflearn.fully_connected(data_s, hidden_data_layer_nodes,regularizer='L2', weight_decay=0.01)#
            fc1_1 = tflearn.fully_connected(path_fc2, hidden_layer_nodes1,regularizer='L2', weight_decay=0.01)#from state
            fc1_2 = tflearn.fully_connected(data_s, hidden_layer_nodes1,regularizer='L2', weight_decay=0.01)#from analytic action
            fc1 = tflearn.activation(tf.matmul(path_fc2, fc1_1.W) + tf.matmul(data_s, fc1_2.W) + fc1_1.b + fc1_2.b, activation='relu')
        else:
            fc1 = tflearn.fully_connected(state, hidden_layer_nodes1,regularizer='L2', weight_decay=0.01)#from state
            #fc1 = tflearn.layers.normalization.batch_normalization(fc1)
            fc1 = tflearn.activations.relu(fc1)

            #fc3 = tflearn.fully_connected(fc1, hidden_layer_nodes3,regularizer='L2', weight_decay=0.01)
            #fc3 = tflearn.activations.relu(fc3)

        # Add the action tensor in the 2nd hidden layer
        # Use two temp layers to get the corresponding weights and biases
        fc2_1 = tflearn.fully_connected(fc1, hidden_layer_nodes2,regularizer='L2', weight_decay=0.01)
        fc2_2 = tflearn.fully_connected(action, hidden_layer_nodes2,regularizer='L2', weight_decay=0.01)

        fc2 = tflearn.activation(tf.matmul(fc1, fc2_1.W) + tf.matmul(action, fc2_2.W)  + fc2_2.b + fc2_1.b, activation='relu')#
          
        #fc3 = tflearn.fully_connected(fc2, hidden_layer_nodes3,regularizer='L2', weight_decay=0.01)
        #fc3 = tflearn.activations.relu(fc3)

        # linear layer connected to 1 output representing Q(s,a)
        # Weights are init to Uniform[-3e-3, 3e-3]
        init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)

        Qa = tflearn.fully_connected(fc2, 1, weights_init=init,bias_init = init, regularizer='L2', weight_decay=0.01)
        return Qa


    def get_Qa(self,s,a):#get Q(s,a) at a specific state
        return self.sess.run(self.Qa, feed_dict={self.state: s,self.action_in: a})
        
    def get_targetQa(self,s,a):#get Q(s,a) at a specific state
        return self.sess.run(self.target_Qa, feed_dict={self.state: s,self.action_in: a})
       
    def get_actions(self,s):
        return self.sess.run(self.action_out, feed_dict = {self.state: s})

    def get_target_actions(self,s):
        return self.sess.run(self.target_action_out, feed_dict = {self.state: s})

    def get_actor_parms(self):
        return self.sess.run(self.actor_params)

    def get_actor_grads(self,s,a):
        return self.sess.run(self.actor_gradients, feed_dict = {self.state: s,self.action_in: a})
        
    def get_norm_actor_grads(self,s,a):
        return self.sess.run(self.norm_actor_gradients, feed_dict = {self.state: s,self.action_in: a})

    def get_neg_Q_grads(self,s,a):
        return self.sess.run(self.Q_grads_neg, feed_dict = {self.state: s,self.action_in: a})

    def get_critic_loss(self,s,a,targetQa_in):
        return self.sess.run(self.Q_loss, feed_dict={self.state: s ,self.action_in: a ,self.targetQa_in: targetQa_in})# 
         
    def Update_critic(self,s,a,targetQa_in):
        self.sess.run(self.update_critic, feed_dict={self.state: s ,self.action_in: a ,self.targetQa_in: targetQa_in})# 
        return 

    def Update_actor(self,s,a):
        self.sess.run(self.update_actor, feed_dict={self.state: s ,self.action_in: a})# 
        return
    def update_targets(self):
        for var in self.update_actor_target_vec:
            self.sess.run(var)
        for var in self.update_critic_target_vec:
            self.sess.run(var)
    def copy_targets(self):
        for var in self.copy_actor_target_vec:
            self.sess.run(var)
        for var in self.copy_critic_target_vec:
            self.sess.run(var)

    ###########################
    #analytic:
    def Update_analytic_actor(self,s,analytic_action):
        self.sess.run(self.update_analytic_actor, feed_dict={self.state: s ,self.analytic_action: analytic_action})# 
        return 
    def get_analytic_actor_loss(self,s,analytic_action):
        return self.sess.run(self.analytic_actor_loss, feed_dict={self.state: s ,self.analytic_action: analytic_action})# 
         

   


    ###########################
    def init_summaries(self):
        reward = tf.Variable(0.)
        tf.summary.scalar("Reward", reward)
        self.summaries_list = [reward]
        return 
    def update_summaries(self,reward):
        self.sess.run(self.merged, feed_dict={self.summaries_list[0]: reward})

    #def __del__(self):
    #    self.sess.close()


        