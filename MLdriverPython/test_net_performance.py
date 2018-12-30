from model_based_net import model_based_network
import agent_lib as pLib
import hyper_parameters
import matplotlib.pyplot as plt
import numpy as np
import library as lib
import enviroment1

def save_Y_Y_(file_name,Y,Y_):
    with open(file_name, 'w') as f:#append data to the file
        for y, y_ in zip(Y,Y_):
            f.write("%s\t %s\t\n" % (y,y_))
def plot_distribution(data,name):
    plt.figure(name+" distribution")
    plt.hist(data,bins='auto')
    print("variance of",name,":",np.var(data))
def plot_comparison(real, predicted,name):
    plt.figure(name)
    plt.plot(real,'o')
    plt.plot(predicted,'o')

def split_to_columns(arr2d,n):
    arr1 = []
    arr2 = []
    for a in arr2d:
        arr1.append(a[:n])
        arr2.append(a[n:])
    return arr1,arr2


def convert_data(Replay,envData):
    rand_state, rand_a, rand_next_state, end,fail= map(np.array, zip(*Replay.memory))
    X,Y_ = envData.create_XY_(rand_state,rand_a,rand_next_state)
    return X,Y_,end,fail

def train_net(HP,net,Replay,envData,waitFor,num_train):
    losses = []
    train_count = 0
    plt.ion()
    for i in range(num_train):
        if waitFor.stop == [True]:
            break
        #with trainShared.Lock:
        rand_state, rand_a, rand_next_state, rand_end,_ = Replay.sample(HP.batch_size)
        #update neural networs:
        pLib.model_based_update(rand_state, rand_a, rand_next_state,rand_end,net,HP,envData)
        train_count+=1
        if train_count % 100 == 0:
            #X,Y_,end,fail = convert_data(ReplayTest)
            X,Y_ = envData.create_XY_(rand_state, rand_a, rand_next_state)
            loss = float(net.get_loss(X,Y_))
            if loss > 10:
                print("what??")
            losses.append(loss)
            plt.cla()
            plt.plot(losses)
            plt.plot(lib.running_average(losses,50))
            plt.draw()
            plt.pause(0.0001)
            print("train:",train_count, "loss:",loss)
    plt.ioff()        
    plt.show()
    net.save_model(HP.save_file_path)

def comp_abs_states(init_sample,pred_vec):
    pos = [0,0]
    ang = 0
    roll = 0
    states = [[pos[0],pos[1],ang,init_sample[0],init_sample[1],roll]]#pos ang vel steer 
    for pred in pred_vec:
        pos = lib.to_global(pred[:2], pos, ang)
        ang = pred[2] + ang#
        states.append([pos[0],pos[1],ang,pred[3],pred[4],pred[5]])
        
    return states

def predict_n_next(net,X,end,n):
    pred_vec = []
    sample = list(X[0])
    if end[0] == True:
        return pred_vec
    for i in range(1,n):#n times 
        if end[i] == True:
            break
        #print(sample)
        prediction = net.get_Y([sample])[0]#predict_next(features_num,train_data, sample, k, p)
        pred_vec.append(prediction)#x,y,ang,vel, steer - all relative 
        sample[0] = prediction[3]#vel
        sample[1] = prediction[4]#steer
        sample[2] = X[i][2]#action steer
        sample[3] = X[i][3] #action acceleration
        #print(prediction)
    return pred_vec

def compare_n_samples(net,X,end,Y_,n):
    pred_vec = predict_n_next(net,X,end,n)
    abs_pred = comp_abs_states(X[0],pred_vec)    
    abs_real_pred = comp_abs_states(X[0],Y_[:len(pred_vec)])
    return abs_pred,abs_real_pred

if __name__ == "__main__": 
    restore = True
    train = True
    split_buffer = True
    train_part = 0.5
    num_train = 1000000

    waitFor = lib.waitFor()
    HP = hyper_parameters.ModelBasedHyperParameters()
    envData = enviroment1.OptimalVelocityPlannerData('model_based')
    net = model_based_network(envData.X_n,envData.Y_n,HP.alpha,envData.observation_space.range)

    Replay = pLib.Replay(1000000)
    #velocity, steering angle, steering action, acceleration action,  rel x, rel y, rel ang, velocity next, steering next, roll
    Replay.restore(HP.restore_file_path)
    print("lenght of buffer: ",len(Replay.memory))
    ReplayTrain = pLib.Replay(1000000)
    ReplayTest = pLib.Replay(1000000)
    if split_buffer:
        ReplayTrain.memory = Replay.memory[int(train_part*len(Replay.memory)):]#Replay.memory[:int(train_part*len(Replay.memory))]
        ReplayTest.memory = Replay.memory[:int(train_part*len(Replay.memory))]#Replay.memory[int(train_part*len(Replay.memory)):]#

    if restore:
        net.restore(HP.restore_file_path)

    if train:
        train_net(HP,net,ReplayTrain,envData, waitFor,num_train)

    
    X,Y_,end,fail = convert_data(ReplayTest,envData)
    print("test loss: ",net.get_loss(X,Y_))
    #compare one step prediction:
   ## Y = np.array([net.get_Y(x) for x in X])
    Y = net.get_Y(X)
    Y = np.array(Y)
    Y_ = np.array(Y_)
    errors = Y - Y_

    save_Y_Y_("data1.txt",Y,Y_)


    ##compare n step prediction:
    #vec_Y_n =  []
    #vec_Y_n_ = []
    #n = 2
    #for i in range(len(X)-n):
    #    Y_n, Y_n_ = compare_n_samples(net,X[i:i+n],end[i:i+n],Y_[i:i+n],n)#compute the predicted and real state during n steps 
    #    if  len(Y_n) < n:
    #        continue
    #    vec_Y_n_.append(Y_n_[-1])
    #    vec_Y_n.append(Y_n[-1])

    #vec_Y_n = np.array(vec_Y_n)
    #vec_Y_n_ = np.array(vec_Y_n_)
    #errors = vec_Y_n - vec_Y_n_

    #fail_Y, fail_Y_ = [],[]
    #for y, y_ in zip(vec_Y_n,vec_Y_n_):
    #    if abs(y[5]) < envData.max_plan_roll and abs(y_[5]) > envData.max_plan_roll:
    #        fail_Y.append(y[5])
    #        fail_Y_.append(y_[5])
    #total_fails = 0
    #for roll in vec_Y_n_[:,5]:
    #    if roll > envData.max_plan_roll:
    #        total_fails+=1 
    #print("total_fails:",total_fails)
    #print("fails: ",len(fail_Y))
    #plot_comparison(fail_Y_, fail_Y,"fails")

    #plot_distribution(errors[:,0],"error x")
    #plot_distribution(errors[:,1],"error y")
    #plot_distribution(errors[:,2],"error ang")
    #plot_distribution(errors[:,3],"error vel")
    #plot_distribution(errors[:,5],"error roll")
    ##plot_comparison(Y[:,0], Y_[:,0],"x")
    #plot_comparison(vec_Y_n_[:,0], vec_Y_n[:,0],"ang")
    #plot_comparison(vec_Y_n_[:,1], vec_Y_n[:,1],"vel")
    #plot_comparison(vec_Y_n_[:,2], vec_Y_n[:,2],"roll")
    #plot_comparison(vec_Y_n_[:,3], vec_Y_n[:,3],"x")
    #plot_comparison(vec_Y_n_[:,5], vec_Y_n[:,5],"y")

    #plot_comparison(Y_[:,0], Y[:,0],"vel")
    #plot_comparison(Y_[:,1], Y[:,1],"steer")
    #plot_comparison(Y_[:,2], Y[:,2],"roll")
    #plot_comparison(Y_[:,3], Y[:,3],"x")
    #plot_comparison(Y_[:,4], Y[:,4],"y")
    #plot_comparison(Y_[:,5], Y[:,5],"ang")

    
    #plot_comparison(Y_[:,0], Y[:,0],"roll")
    #plot_distribution(errors[:,0],"error roll")

    plot_comparison(Y_[:,0], Y[:,0],"vel0")
    plot_comparison(Y_[:,1], Y[:,1],"vel1")
    plot_comparison(Y_[:,2], Y[:,2],"vel2")
    plot_comparison(Y_[:,3], Y[:,3],"ang_vel0")
    plot_comparison(Y_[:,4], Y[:,4],"ang_vel1")
    plot_comparison(Y_[:,5], Y[:,5],"ang_vel2")
    plot_comparison(Y_[:,6], Y[:,6],"acc0")
    plot_comparison(Y_[:,7], Y[:,7],"acc1")
    plot_comparison(Y_[:,8], Y[:,8],"acc2")
    plot_comparison(Y_[:,9], Y[:,9],"ang_acc0")
    plot_comparison(Y_[:,10], Y[:,10],"ang_acc1")
    plot_comparison(Y_[:,11], Y[:,11],"ang_acc2")
    plot_comparison(Y_[:,12], Y[:,12],"steer")
    plot_comparison(Y_[:,13], Y[:,13],"roll")
    plot_comparison(Y_[:,14], Y[:,14],"rear")
    plot_comparison(Y_[:,15], Y[:,15],"front")
    plot_comparison(Y_[:,16], Y[:,16],"dx")
    plot_comparison(Y_[:,17], Y[:,17],"dy")
    plot_comparison(Y_[:,18], Y[:,18],"dang")

    #plot_comparison(Y_[:,14], Y[:,14],"dx")
    #plot_comparison(Y_[:,15], Y[:,15],"dy")
    #plot_comparison(Y_[:,16], Y[:,16],"dang")

    plt.show()

       