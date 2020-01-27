import threading


class compError(threading.Thread):
    def __init__(self,Agent,done):
        self.Agent = Agent
        self.done = done
        threading.Thread.__init__(self)


    def run(self):
        self.Agent.update_episode_var()
        self.done[0] = True



