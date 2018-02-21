import numpy as np
from ReinforceNet import Network
import math
state = [1.,2.,3,4]
net = Network(1e-6,len(state),2) 



Q_corrected = np.copy(Q)
Q_corrected[0][0] = 10000
print("Q - corrected: ",Q_corrected)
net.update_sess(state,Q_corrected)
Q = net.get_Q(state)
print("Q:",Q)

