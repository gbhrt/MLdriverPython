
class HyperParameters:
    def __init__(self):
        self.epsilon_start = 1.0
        self.epsilon = 0.1
        self.gamma = 0.99
        self.tau = 0.001 #how to update target network compared to Q network
        self.num_of_runs = 100000
        self.alpha_actor = 0.0001# for Pi 1e-5 #learning rate
        self.alpha_critic = 0.001#for Q
        self.batch_size = 64
        self.replay_memory_size = 100000
        self.num_of_TD_steps = 15 #for TD(lambda)

        #########################
        self.render_flag = True
        self.plot_flag = True
        self.restore_flag = True
        self.skip_run = False
        self.reset_every = 3
        self.save_every = 1000
        self.seed = 1234
        self.save_name = "model30" #model6.ckpt - constant velocity limit - good. model7.ckpt - relative velocity.
        #model10.ckpt TD(5) dt = 0.2 alpha 0.001 model13.ckpt - 5 points 2.5 m 0.001 TD 15
        #model8.ckpt - offline trained after 5 episode - very clear
        self.restore_name = "model29" # model2.ckpt - MC estimation 
        self.run_data_file_name = 'running_record1'