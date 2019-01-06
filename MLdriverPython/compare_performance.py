import numpy as np
import os
import json
import matplotlib.pyplot as plt

def read_data(file_name):
    with open(file_name, 'r') as f:#append data to the file
        data = json.load(f)
    return data

def convert(data):
    return data[0],data[1],data[2],data[3],data[4],data[5],data[6]
def analyse_data(data_vec):#get 
    names,train_losses,test_losses = [],[],[]
    for data in data_vec:
        data_name,train_X,train_Y, train_Y_,test_X,test_Y, test_Y_ = data[0],data[1],data[2],data[3],data[4],data[5],data[6]
        names.append(data_name)
        train_losses.append(net.get_loss(train_X,train_Y_))
        test_losses.append(net.get_loss(test_X,test_Y_))

    for name,train_loss,test_loss in zip(names,train_loses,test_loses):
        print(name,train_loss,test_loss)


#from model_based_net import model_based_network

#models_list = ["test1","test2"]
#data_set = models_list
#net = model_based_network(envData.X_n,envData.Y_n,HP.alpha,envData.observation_space.range)
def comp_var(X,Y,Y_,feature_name):
    real,predicted = [],[]
    for y_,y in zip(Y_,Y):
            real.append(y_[feature_name])
            predicted.append(y[feature_name])

    error = np.array(real) - np.array(predicted)
    return np.var(error)

import enviroment1
envData = enviroment1.OptimalVelocityPlannerData('model_based')
folder = os.getcwd()+"\\files\\train_data\\"

file_name_list = [folder+"data3.txt",folder+"data4.txt"]
data_vec = []
for file_name in file_name_list:
    data_vec.append(read_data(file_name))

models =[]
test_var_vec,train_var_vec = [],[]

features = []

for data in data_vec:
    
    test_var,train_var = [],[]
    data_name,train_X,train_Y, train_Y_,test_X,test_Y, test_Y_ = convert(data)
    models.append(data_name)
    for key, value in train_Y_[0].items():
        if key not in features:
            features.append(key)
    for name in features:
        test_var.append(comp_var(test_X,test_Y, test_Y_,name))
        train_var.append(comp_var(train_X,train_Y, train_Y_,name))

    test_var_vec.append(test_var)
    train_var_vec.append(train_var)
    print(test_var)
    print(train_var)


col_labels = []#"features"
for i in range(len(models)):
    col_labels.append("test_var "+str(i))
    col_labels.append("train_var "+str(i))
cell_text = []
for i in range(len(features)):
    row = []
    for j in range(len(models)):
        row.append(test_var_vec[j][i])
        row.append(train_var_vec[j][i])
    cell_text.append(row)


fig, ax = plt.subplots(2,figsize=(10,8))
ax[0].axis('off')
ax[1].axis('off')
ax[0].table(cellText=cell_text,
                      rowLabels=features,
                      colLabels=col_labels,
                       loc='center'
                      )
cell_text = []
for i in range(len(models)): 
    cell_text.append([i,data_vec[i][0]])

ax[1].table(cellText=cell_text,
                      colLabels=["model number","description"],
                       loc='center'
                      )
plt.show()