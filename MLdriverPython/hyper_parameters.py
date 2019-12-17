import os
class HyperParameters:
    def __init__(self):
        self.gui_flag = True
        self.env_mode = 'DDPG'
        #self.epsilon_start = 1.0
        #self.epsilon = 0.1
        self.gamma = 0.99
        self.tau = 0.001 #how to update target network compared to Q network
        self.num_of_runs = 100000
        self.alpha_actor = 0.0001# for Pi 1e-5 #learning rate
        self.alpha_critic = 0.001#for Q 0.001
        self.alpha_analytic_actor = 1e-5#1e3 - too high, loss stop at 0.1
        self.alpha_analytic_critic = 1e-4
        self.batch_size = 64
        self.replay_memory_size = 1000000
        self.train_num = 2# how many times to train in every step
        self.sample_ratio = 1.0
        self.epsilon = 0.1
        #########################
        self.evaluation_flag = False
        self.reduce_vel = 0.0
        self.add_feature_to_action  = False
        self.analytic_action = False
        self.train_flag =True
        self.noise_flag = True
        self.always_no_noise_flag = False
        self.evaluation_every = 1000000
        self.test_same_path = False
        self.run_same_path = False
        self.conv_flag = False
        self.gym_flag = False
        self.render_flag = True
        self.plot_flag = True
        self.restore_flag = False
        self.skip_run = False
        self.reset_every = 3
        self.save_every = 50
        self.save_every_train_number = 2500
        self.seed = [1111]#,1112,1113,1114,1115]
        self.save_name ="REVO"
        self.folder_path = os.getcwd()+ "/files/models/paper_fix/"
        self.save_file_path = self.folder_path+self.save_name+"/"

        self.restore_name = "REVO"
        self.restore_file_path = self.folder_path+self.restore_name+"/"

        if self.always_no_noise_flag:
            self.noise_flag = False

class SafteyHyperParameters:
    def __init__(self):
        self.gui_flag = True
        #self.epsilon_start = 1.0
        #self.epsilon = 0.1
        self.gamma = 0.99
        self.tau = 0.001 #how to update target network compared to Q network
        self.num_of_runs = 100000
        self.alpha_actor = 0.0001# for Pi 1e-5 #learning rate
        self.alpha_critic = 0.001#for Q 0.001
        self.alpha_analytic_actor = 1e-5#1e3 - too high, loss stop at 0.1
        self.alpha_analytic_critic = 1e-4
        self.batch_size = 64
        self.replay_memory_size = 1000000
        self.train_num = 2# how many times to train in every step
        self.sample_ratio = 1.0
        self.epsilon = 0.1
        self.minQ = -0.5
        #########################
        self.env_mode = "SDDPG"#SDDPG DDPG_target SDDPG_pure_persuit
        self.evaluation_flag = False
        self.reduce_vel = 0.0
        self.add_feature_to_action  = False
        self.analytic_action = False
        self.train_flag =True
        self.noise_flag = True
        self.always_no_noise_flag = False
        self.evaluation_every = 10
        self.test_same_path = False
        self.run_same_path = False
        self.conv_flag = True
        self.gym_flag = False
        self.render_flag = True
        self.plot_flag = True
        self.stabilize_flag = False
        self.constant_velocity = None #5.0
        self.DQN_flag = False
        #self.pure_persuit_flag = False
        self.restore_flag = True  
        self.skip_run = False
        self.reset_every = 3
        self.save_every = 100
        self.save_every_train_number = 25000000
        self.seed = [1111]#,1112,1113,1114,1115]
        self.save_name ="SDDPG1"#SDDPG_vel_and_steer_roll_reward2- roll reward, no saftey, ~0.8 of VOD 20-30% fails 
        self.folder_path = os.getcwd()+ "/files/models/new_state/"
        #SDDPG_vel_and_steer_roll_reward3 -with roll feature, doesn't converge
        self.save_file_path = self.folder_path+self.save_name+"/"

        self.restore_name = "SDDPG1"#SDDPG_pure_persuit saftey good but limit velocity because reward to low. SDDPG_pure_persuit1 - good
        #SDDPG_pure_persuit3 - conv_flag, path layer sizes 50,20
        #SDDPG_pure_persuit3 - conv_flag, path layer sizes 20,5
        self.restore_file_path = self.folder_path+self.restore_name+"/"

        if self.always_no_noise_flag:
            self.noise_flag = False

class ModelBasedHyperParameters:#global settings of the program.
    def __init__(self):
       
        self.program_mode =  "train_in_env"#"test_net_performance" train_in_env, test_actions  timing
        if self.program_mode == "test_net_performance" or self.program_mode == "test_actions":
            self.gui_flag = False
        else:
            self.gui_flag = True
        self.MF_policy_flag = False
        self.emergency_action_flag = False
        self.emergency_steering_type = 4#1 - stright, 2 - 0.5 from original steering, 3-steer net, 4-same steering, 5-roll proportional
        self.direct_stabilize = True  
        #########################
        #for the shared main:
        self.env_mode = "model_based"
        self.evaluation_flag = False
        self.always_no_noise_flag = False
        self.reduce_vel = 0.0
        self.num_of_runs = 100000
        self.save_every_train_number = 25000000
        self.evaluation_every = 999999999
        self.add_feature_to_action = False
        #####################
        self.pause_for_training = False
        self.max_steps = 200000
        self.train_flag = True
        #self.noise_flag = True
        #self.always_no_noise_flag = False
        self.analytic_action = False
       # self.zero_noise_every = 1
        self.evaluation_every = 10
        self.test_same_path = False
        self.run_same_path = False
       # self.conv_flag = False
        self.gym_flag = False
        self.render_flag = True
        self.plot_flag = True
        self.restore_flag = False
        self.skip_run = False
        self.reset_every = 1
        self.save_every = 10000000000
        self.save_every_time = 5000 #minutes
        self.seed = [1111]
        self.net_name = "tf_model"
        self.save_name = "MB_test1"#"MB_R_long2" MB_R_DS1
        self.folder_path = os.getcwd()+ "/files/models/model_based/"
        #self.save_file_path = os.getcwd()+ "/files/models/model_based/"+self.save_name+"/"f
        self.save_file_path = self.folder_path +self.save_name+"/"

        self.restore_name = "MB_test1"#"MB_R_long1"#MB_R_long2
        #self.restore_file_path = os.getcwd()+ "/files/models/model_based/"+self.restore_name+"/"
        self.restore_file_path = self.folder_path +self.restore_name+"/"

       
