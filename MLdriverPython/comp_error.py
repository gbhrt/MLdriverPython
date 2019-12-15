import threading


class compError(threading.Thread):
    def __init__(self,Agent,step_count,done):
        self.Agent = Agent
        self.step_count = step_count
        self.done = done
        threading.Thread.__init__(self)


    def run(self):
        self.Agent.update_episode_var(self.step_count)
        self.done[0] = True



