import numpy as np

def read_Y_Y_(file_name):
    with open(file_name, 'r') as f:#append data to the file
        data = f.readlines()
        data = [x.strip().split() for x in data]
        Y,Y_ = [],[]
        for x in data:
            Y.append(float(x[0]))
            Y_.append(float(x[1]))
    return Y,Y_



