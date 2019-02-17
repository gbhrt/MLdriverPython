import tkinter as tk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import math
import time

def rotate_vec(vec,ang):
    rvec =np.array([0.,0.])
    rvec[0] = math.cos(ang)*vec[0] - math.sin(ang)*vec[1]
    rvec[1] = math.sin(ang)*vec[0] + math.cos(ang)*vec[1]
    return rvec

def plot_velocities(fig):#,path):
    #if len(self.real_path.time) > 0:
    
    #max_time_ind = len(path.time)
    #fig.clear()
    lines = []

    lines.append(fig.plot([],[],label = "velocity limit")[0])
    lines.append(fig.plot([],[],label = "analytic velocity")[0])
    lines.append(fig.plot([],[],label = "vehicle velocity")[0])
    lines.append(fig.plot([],[],'.',color = 'red',label = "emergency action")[0])
    #plt.legend()
    return lines

def update_velocities(lines,path,vec_emergency_action,index):
    #lines[0].set_data(np.array(path.distance)[:index],np.array(path.analytic_velocity_limit)[:index])
    
    #lines[1].set_data(np.array(path.distance)[:index],np.array(path.analytic_velocity)[:index])
    #lines[2].set_data(path.distance,path.velocity)
    lines[0].set_data(np.array(path.time)[:index],np.array(path.analytic_velocity_limit)[:index])   
    lines[1].set_data(np.array(path.time)[:index],np.array(path.analytic_velocity)[:index])
    lines[2].set_data(path.time[:index],path.velocity[:index])
    x,y=[],[]
    for t,emergency_a in zip(path.time,vec_emergency_action):
        if emergency_a:
            x.append(t)
            y.append(0)
    #lines[3].set_data(x[:index],y[:index])
    lines[3].set_data(x,y)

class data_plots():
    def __init__(self,root,spinBox,guiShared,index):
        self.guiShared = guiShared
        self.root = root
        self.spinBox = spinBox
        self.index = index
        self.last_index = -1

        #self.figure1 = plt.Figure(figsize=(6,5))#, dpi=100)
        #self.ax1 = self.figure1.add_subplot(111)
        #self.ax2 = self.figure1.add_subplot(121)
        #self.figure1, (self.ax1,self.ax2,self.ax3) = plt.subplots(3, 1,figsize=(5,7))#
        self.figure1, (self.ax1,self.ax2,self.ax3) = plt.subplots(3, 1,figsize=(5,3))#

        #self.dataManager.roll = [0,1]
        self.ax1.set_ylim(-0.1,0.1)
        self.ax1.set_xlim(0,self.guiShared.max_time)
        self.ax2.set_ylim(0,25)
        self.ax2.set_xlim(0,self.guiShared.max_time)
        self.ax3.set_ylim(-0.2,0.2)
        self.ax3.set_xlim(0,12)
        
        
        #self.line1 = self.ax1.plot(np.array(self.dataManager.roll))[0]
        
        #self.line_roll_var1 = self.ax3.plot(np.array(self.dataManager.vec_planned_roll[-1])+0.1,color = "red")[0]
        #self.line_roll_var2 = self.ax3.plot(np.array(self.dataManager.vec_planned_roll[-1])-0.1,color = "red")[0]
        #self.line_roll_mean = self.ax3.plot(np.array(self.dataManager.vec_planned_roll[-1]),color = "green",label = "roll")[0]
        #self.line_emergency_roll_var1 = self.ax3.plot(np.array(self.dataManager.vec_emergency_planned_roll[-1])+0.1,color = "red")[0]
        #self.line_emergency_roll_var2 = self.ax3.plot(np.array(self.dataManager.vec_emergency_planned_roll[-1])-0.1,color = "red")[0]
        #self.line_emergency_roll_mean = self.ax3.plot(np.array(self.dataManager.vec_emergency_planned_roll[-1]),color = "blue",label = "emergency roll")[0]
        #self.ax3.fill_between(np.arange(15),self.guiShared.max_roll*np.ones(15),-self.guiShared.max_roll*np.ones(15),color = "#dddddd" )

        self.line1 = self.ax1.plot([])[0]
        
        self.line_roll_var1 = self.ax3.plot([],color = "red")[0]
        self.line_roll_var2 = self.ax3.plot([],color = "red")[0]
        self.line_roll_mean = self.ax3.plot([],color = "green",label = "roll")[0]
        self.line_emergency_roll_var1 = self.ax3.plot([],color = "red")[0]
        self.line_emergency_roll_var2 = self.ax3.plot([],color = "red")[0]
        self.line_emergency_roll_mean = self.ax3.plot([],color = "blue",label = "emergency roll")[0]
        self.ax3.fill_between(np.arange(15),self.guiShared.max_roll*np.ones(15),-self.guiShared.max_roll*np.ones(15),color = "#dddddd" )

        #self.ax3.legend()
        self.lines = plot_velocities(self.ax2)#,self.dataManager)
        self.ax1.grid(True)
        self.ax2.grid(True)
        self.ax3.grid(True)
        #self.figure2, self.ax1 = plt.subplots(1, 1,figsize=(5,4))


        self.canvas = FigureCanvasTkAgg(self.figure1, master= root)
        #self.canvas.get_tk_widget().pack()#side=tk.LEFT, fill=tk.BOTH)
        self.canvas.get_tk_widget().grid(row = 0,column = 0, columnspan=2)
        
        self.plot_data()

    def plot_data(self):
        if len(self.guiShared.planningData.vec_planned_roll)>0:
            path = self.guiShared.real_path
            #self.dataManager.roll = [1,3,2]
            #plt.cla()
            #self.ax1.clear()
            #self.ax1.plot(self.dataManager.roll)
            #print("roll:",self.dataManager.roll)
            #print("len roll:",len(self.dataManager.roll))
            #print("dis:",path.distance)
            #self.ax3.collections.clear()
            #self.ax3.fill_between(np.arange(len(self.dataManager.planned_roll)),np.array(self.dataManager.planned_roll)+2,np.array(self.guiShared.planned_roll)-2)
            if self.index[0] >=0:
                self.index[0] = min(int(self.spinBox.get()),len(self.guiShared.roll)-1)

            if self.index[0]!=self.last_index:
                
                print("-------------------------------------------------")
                print("index: ",self.index[0])
                if self.guiShared.planningData.vec_emergency_action[self.index[0]]:
                    print("emergency action")
                print("planned roll:\n",self.guiShared.planningData.vec_planned_roll[self.index[0]])
                print("emergency planned roll:\n",self.guiShared.planningData.vec_emergency_planned_roll[self.index[0]])
                print("planned velocity:\n",self.guiShared.planningData.vec_planned_vel[self.index[0]])
                print("emergency planned velocity:\n",self.guiShared.planningData.vec_emergency_planned_vel[self.index[0]])
                print("planned steer:\n",self.guiShared.planningData.vec_planned_steer[self.index[0]])
                print("emergency planned steer:\n",self.guiShared.planningData.vec_emergency_planned_steer[self.index[0]])
                print("planned acc:\n",self.guiShared.planningData.vec_planned_acc[self.index[0]])
                print("emergency planned acc:\n",self.guiShared.planningData.vec_emergency_planned_acc[self.index[0]])
                self.last_index = self.index[0]

            

                


            self.line_roll_mean.set_data(np.arange(len(self.guiShared.planningData.vec_planned_roll[self.index[0]])),np.array(self.guiShared.planningData.vec_planned_roll[self.index[0]]))
            self.line_roll_var1.set_data(np.arange(len(self.guiShared.planningData.vec_planned_roll[self.index[0]])),np.array(self.guiShared.planningData.vec_planned_roll[self.index[0]])-np.array(self.guiShared.planningData.vec_planned_roll_var[self.index[0]]))
            self.line_roll_var2.set_data(np.arange(len(self.guiShared.planningData.vec_planned_roll[self.index[0]])),np.array(self.guiShared.planningData.vec_planned_roll[self.index[0]])+np.array(self.guiShared.planningData.vec_planned_roll_var[self.index[0]]))
        
            self.line_emergency_roll_mean.set_data(np.arange(len(self.guiShared.planningData.vec_emergency_planned_roll[self.index[0]])),np.array(self.guiShared.planningData.vec_emergency_planned_roll[self.index[0]]))
            self.line_emergency_roll_var1.set_data(np.arange(len(self.guiShared.planningData.vec_emergency_planned_roll[self.index[0]])),np.array(self.guiShared.planningData.vec_emergency_planned_roll[self.index[0]])-np.array(self.guiShared.planningData.vec_emergency_planned_roll_var[self.index[0]]))
            self.line_emergency_roll_var2.set_data(np.arange(len(self.guiShared.planningData.vec_emergency_planned_roll[self.index[0]])),np.array(self.guiShared.planningData.vec_emergency_planned_roll[self.index[0]])+np.array(self.guiShared.planningData.vec_emergency_planned_roll_var[self.index[0]]))
        


            if len(path.distance) > 0:
                #if self.index[0] == -1:
                #    max_time_ind = len(path.time)
                #else:
                #    max_time_ind = self.index[0]
                max_time_ind = len(path.time) if self.index[0] == -1 else self.index[0]
                self.line1.set_data(path.time[:max_time_ind],self.guiShared.roll[:max_time_ind])

                update_velocities(self.lines,path,self.guiShared.planningData.vec_emergency_action,self.index[0])
                self.canvas.draw()

        self.root.after(50,self.plot_data)

class draw_state:
    def __init__(self,root,spinBox,guiShared,index):
        self.guiShared = guiShared
        
        self.index = index
        self.root = root
        self.spinBox = spinBox
        
        
        self.scale = 5
        self.canvas_width=250
        self.canvas_height=250
        self.canvas = tk.Canvas(self.root,width = self.canvas_width, height = self.canvas_height)
        self.canvas.config(background="white")
        #self.canvas.pack()
        self.canvas.grid(row = 1,rowspan = 2)
        self.vehicle_pos = np.array([self.canvas_width*0.5,self.canvas_height*0.85])

        self.draw()


        
    def draw(self):
        if self.index[0] >=0:
            self.index[0] = min(int(self.spinBox.get()),len(self.guiShared.planningData.vec_planned_roll)-1)
        self.canvas.delete("all")
        if self.guiShared.steer is not None:
            self.draw_vehicle(self.guiShared.steer)
        if len(self.guiShared.planningData.vec_path)>0 and len(self.guiShared.planningData.vec_path[self.index[0]].position)>1:
            self.draw_path(self.guiShared.planningData.vec_path[self.index[0]].position,color = "red")
        if len(self.guiShared.planningData.vec_predicded_path) > 0 and len(self.guiShared.planningData.vec_predicded_path[self.index[0]])>1:
            self.draw_path(self.guiShared.planningData.vec_predicded_path[self.index[0]],color = "green")
        if len(self.guiShared.planningData.vec_emergency_predicded_path) > 0 and len(self.guiShared.planningData.vec_emergency_predicded_path[self.index[0]])>1:
            self.draw_path(self.guiShared.planningData.vec_emergency_predicded_path[self.index[0]],color = "blue")

        self.root.after(50,self.draw)
        #self.draw_vehicle(0.5)
        #test_path = [[0,0],[1,1],[2,4],[5,7],[7,7]]
        #self.draw_path(test_path)
    def draw_vehicle(self,ang):
        width = 2.08*self.scale
        length = 4.0*self.scale
        #self.cords.translate([self.canvas_width*0.5,self.canvas_height*0.75])
        #self.canvas.create_rectangle(0,0, 100,100,fill = "black")

        #body
        self.canvas.create_rectangle(self.vehicle_pos[0] -width/2,self.vehicle_pos[1] -length/2,self.vehicle_pos[0] + width/2,self.vehicle_pos[1] +length/2,fill = "blue")
        
        wheel_width = 0.5*self.scale
        wheel_height = 1.0*self.scale
        space = 0.07*self.scale
        #rear wheels
        wheel_pos = [self.vehicle_pos[0] - (width/2 + space),self.vehicle_pos[1] +length/2]#left 
        self.canvas.create_rectangle(wheel_pos[0] -wheel_width/2,wheel_pos[1] -wheel_height/2,
                                     wheel_pos[0] + wheel_width/2,wheel_pos[1] +wheel_height/2,
                                     fill = "black")
        wheel_pos = [self.vehicle_pos[0] + (width/2 + space),self.vehicle_pos[1] +length/2]#right
        
        self.canvas.create_rectangle(wheel_pos[0] -wheel_width/2,wheel_pos[1] -wheel_height/2,
                                     wheel_pos[0] + wheel_width/2,wheel_pos[1] +wheel_height/2,
                                     fill = "black")
        #front wheels
        wheel_pos = np.array([self.vehicle_pos[0] - (width/2 + space),self.vehicle_pos[1] -length/2])#left 
        p1 = wheel_pos+rotate_vec([-wheel_width/2,-wheel_height/2],ang)
        p2 = wheel_pos+rotate_vec([wheel_width/2,-wheel_height/2],ang)
        p3 = wheel_pos+rotate_vec([wheel_width/2,wheel_height/2],ang)
        p4 = wheel_pos+rotate_vec([-wheel_width/2,wheel_height/2],ang)
        self.canvas.create_polygon([p1[0],p1[1],p2[0],p2[1],p3[0],p3[1],p4[0],p4[1]],
                                     fill = "black")

        wheel_pos = np.array([self.vehicle_pos[0] + (width/2 + space),self.vehicle_pos[1] -length/2])#left 
        
        p1 = wheel_pos+rotate_vec([-wheel_width/2,-wheel_height/2],ang)
        p2 = wheel_pos+rotate_vec([wheel_width/2,-wheel_height/2],ang)
        p3 = wheel_pos+rotate_vec([wheel_width/2,wheel_height/2],ang)
        p4 = wheel_pos+rotate_vec([-wheel_width/2,wheel_height/2],ang)
        self.canvas.create_polygon([p1[0],p1[1],p2[0],p2[1],p3[0],p3[1],p4[0],p4[1]],
                                     fill = "black")


    def draw_path(self,path,color = "black"):
        relative_path = [[pos[0]*self.scale + self.vehicle_pos[0],-pos[1]*self.scale+ self.vehicle_pos[1]] for pos in path]
        flat_path = []
        for pos in relative_path:
            flat_path.append(pos[0])
            flat_path.append(pos[1])
        self.canvas.create_line(flat_path,fill = color)

class TkGui():
    def __init__(self,guiShared):
        self.guiShared = guiShared
        self.index = [-1]
        
        self.root= tk.Tk() 
        screenwidth = self.root.winfo_screenwidth()
        screenheight = self.root.winfo_screenheight()
        self.root.geometry('%dx%d+%d+%d' % (screenwidth*0.4,screenheight*0.9,0,0))#(w, h, x, y)
        self.root.wm_title("Gui")

        
        self.spinBox = tk.Spinbox(self.root, from_=0, to= 120)
        self.spinBox.grid(row = 4, column=1)
        data_plots(self.root,self.spinBox,guiShared,self.index)
        draw_state(self.root,self.spinBox,guiShared,self.index)

        #tk.Button(self.root, text="Quit", command=self.quit).pack()
        tk.Button(self.root, text="Quit", command=self.quit).grid(row = 1,column =1)

        self.pause_btn_text = tk.StringVar()
        tk.Button(self.root,  textvariable = self.pause_btn_text, command=self.pause_after_episode).grid(row = 2, column=1)
        self.pause_btn_text.set("Pause")

        self.replay_btn_text = tk.StringVar()
        tk.Button(self.root,  textvariable = self.replay_btn_text, command=self.replay_mode).grid(row = 3, column=1)
        self.replay_btn_text.set("Set Replay Mode")

        self.root.mainloop()

    def quit(self):
        self.guiShared.request_exit = True
        while not self.guiShared.exit:#wait for finish the program
            time.sleep(0.1)
        self.root.destroy()
    def pause_after_episode(self):
        print("pause at the end of the episode")
        if self.guiShared.pause_after_episode_flag:
            self.guiShared.pause_after_episode_flag = False
            self.pause_btn_text.set("Pause")
        else:
            self.guiShared.pause_after_episode_flag = True
            self.pause_btn_text.set("Continue")

    def replay_mode(self):
        #print("insert index___________________________________________________________________:")
        #inp = input()
        if self.index[0] == -1:
            self.replay_btn_text.set("Exit Replay Mode")

            self.index[0] = 0#
            self.spinBox.config(to =len(self.guiShared.roll)-1) 
        else:
            self.index[0] = -1
            self.replay_btn_text.set("Set Replay Mode")
        return

    

    #def draw_graph(self):
       
    #    self.bar1.draw()
    #    print("roll:--------------------------------",self.dataManager.roll)
        
    #    self.root.after(1000,self.draw_graph())

