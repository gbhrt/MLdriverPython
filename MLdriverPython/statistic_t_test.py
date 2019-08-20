from scipy import stats
import os
import data_manager1

def get_data(folder,names):
    relative_reward_vec = []
    for name in names:#
        restore_path = os.getcwd()+ "/files/models/"+str(folder)+"/"+name+"/"
        restore_name = 'data_manager'
        dataManager = data_manager1.DataManager(restore_path,restore_path,True,save_name = restore_name,restore_name = restore_name)#
        if dataManager.error:
            print("cannot restore dataManager")
            break
        relative_reward_vec.append(dataManager.relative_reward)

    return relative_reward_vec

def save_data(relative_reward_vec,file_name):
    with open(file_name, 'w') as f:#append data to the file
        for relative_reward in relative_reward_vec:
            for relative_reward_i in relative_reward:
                f.write("%s\n" % (relative_reward_i))
            #f.write('\n')
    return

names = ["REVO+A3","REVO+F3","REVO10","VOD_long"]
for name in names:
    relative_reward_vec = get_data("paper_fix_final",[name])#,"REVO+A3","VOD_long"
    save_data(relative_reward_vec,name+"_relative_reward.txt")


##%matplotlib inline

#data = pd.read_csv('data/brain_size.csv', sep=';', na_values=".")
#data['VIQ'].hist()

#stats.ttest_1samp(data['VIQ'], 110)
## pvalue = 0.533, accept null hypothesis
#stats.ttest_1samp(data['VIQ'], 100)
## pvalue = 0.002, reject null hypothesis

## one-tailed, less than 130
#m = 130
#results = stats.ttest_1samp(data['VIQ'], m)
#alpha = 0.05
#if (results[0] < 0) & (results[1]/2 < alpha):
#    print "reject null hypothesis, mean is less than {}".format(m)
#else:
#    print "accept null hypothesis"
    
## one-tailed, greater than 80
#m = 80
#results = stats.ttest_1samp(data['VIQ'], m)
#alpha = 0.05
#if (results[0] > 0) & (results[1]/2 < alpha):
#    print "reject null hypothesis, mean is greater than {}".format(m)
#else:
#    print "accept null hypothesis"