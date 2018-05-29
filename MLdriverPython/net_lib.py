import tensorflow as tf
import os
import pathlib

class NetLib:

    #def update_target_init(self,tau,name):
    #    tvars = tf.trainable_variables()
    #    tvars = [var for var in tvars if name in var.name]
    #    Q_vars = tvars[0:len(tvars)//2]
    #    tarQ_vars = tvars[len(tvars)//2:len(tvars)]
    #    update_var_vec = []
    #    for i in range(len(Q_vars)):
    #        update_var_vec.append(tarQ_vars[i].assign((Q_vars[i].value()*tau) + ((1-tau) * tarQ_vars[i])))
    #    return update_var_vec
    def update_target_init(self,tau,vars, tar_vars):
        update_var_vec = []
        for i in range(len(vars)):
            update_var_vec.append(tar_vars[i].assign((vars[i].value()*tau) + ((1-tau) * tar_vars[i])))
        return update_var_vec
    def copy_target_init(self,tau,vars, tar_vars):
        update_var_vec = []
        for i in range(len(vars)):
            update_var_vec.append(tar_vars[i].assign(vars[i].value()))
        return update_var_vec

    def save_model(self,path,name = "tf_model"):
        try:
                path1 = path +name+"\\"
                pathlib.Path(path1).mkdir(parents=True, exist_ok=True) 
                file_name =  path1+name+".ckpt " #/media/windows-share/MLdriverPython/MLdriverPython/
                saver = tf.train.Saver()
                save_path = saver.save(self.sess, file_name)
                print("Model saved in file: %s" % save_path)
        except:
            print('cannot save - try again')
            self.save_model(path,name)


    def restore(self,path,name = "tf_model"):
        try:
            #name = "model3" #.ckpt
            #if len(args) > 0:
            #    name = args[0]
            #path = os.getcwd()+ "\\models\\policy\\"+name+"\\"#
            path = path +name+"\\"
            file_name =  path+name+".ckpt " #/media/windows-share/MLdriverPython/MLdriverPython/
            # Restore variables from disk.
            saver = tf.train.Saver()
            saver.restore(self.sess, file_name)
            print("Model restored.")
        except:
            print('cannot restore net')
