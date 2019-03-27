
#dynamic model of the vehicle. 
#get state after n steps
#get acc action given next state

class MB_model:
    def __init__(self,net,invNet):
        self.net = net
        self.invNet = invNet
        return 
