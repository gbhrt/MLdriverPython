import tkinter as tk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import math
import time
import copy

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
        self.figure1, (self.ax1,self.ax2,self.ax3,self.ax_episodes) = plt.subplots(4, 1,figsize=(5,3.8))#
        #self.figure1, (self.ax1,self.ax2,self.ax3,self.ax_episodes) = plt.subplots(4, 1,figsize=(7,5))#

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
        self.lines = plot_velocities(self.ax2)
        self.ax1.grid(True)
        self.ax2.grid(True)
        self.ax3.grid(True)
        


        self.canvas = FigureCanvasTkAgg(self.figure1, master= root)
        #self.canvas.get_tk_widget().pack()#side=tk.LEFT, fill=tk.BOTH)
        self.canvas.get_tk_widget().grid(row = 0,column = 0, columnspan=3)
        
        self.plot_data()

    def plot_data(self):
        print_debug_data = False
        if self.index[0] >=0:#update data if spin box changed.
            self.index[0] = min(int(self.spinBox.get()),len(self.guiShared.roll)-1)
            if self.index[0]!=self.last_index:
                self.guiShared.update_data_flag = True
                print_debug_data = True
                self.last_index = self.index[0]

        if self.guiShared.update_data_flag:
            if len(self.guiShared.planningData.vec_planned_roll)>0:
                #self.dataManager.roll = [1,3,2]
                #plt.cla()
                #self.ax1.clear()
                #self.ax1.plot(self.dataManager.roll)
                #print("roll:",self.dataManager.roll)
                #print("len roll:",len(self.dataManager.roll))
                #print("dis:",path.distance)
                #self.ax3.collections.clear()
                #self.ax3.fill_between(np.arange(len(self.dataManager.planned_roll)),np.array(self.dataManager.planned_roll)+2,np.array(self.guiShared.planned_roll)-2)
                if print_debug_data:
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
                    
                if len(self.guiShared.planningData.vec_planned_roll[self.index[0]])>0:
                    self.line_roll_mean.set_data(np.arange(len(self.guiShared.planningData.vec_planned_roll[self.index[0]])),np.array(self.guiShared.planningData.vec_planned_roll[self.index[0]]))
                #self.line_roll_var1.set_data(np.arange(len(self.guiShared.planningData.vec_planned_roll[self.index[0]])),np.array(self.guiShared.planningData.vec_planned_roll[self.index[0]])-np.array(self.guiShared.planningData.vec_planned_roll_var[self.index[0]]))
                #self.line_roll_var2.set_data(np.arange(len(self.guiShared.planningData.vec_planned_roll[self.index[0]])),np.array(self.guiShared.planningData.vec_planned_roll[self.index[0]])+np.array(self.guiShared.planningData.vec_planned_roll_var[self.index[0]]))
                if len(self.guiShared.planningData.vec_emergency_planned_roll[self.index[0]])>0:
                    self.line_emergency_roll_mean.set_data(np.arange(len(self.guiShared.planningData.vec_emergency_planned_roll[self.index[0]])),np.array(self.guiShared.planningData.vec_emergency_planned_roll[self.index[0]]))
                #self.line_emergency_roll_var1.set_data(np.arange(len(self.guiShared.planningData.vec_emergency_planned_roll[self.index[0]])),np.array(self.guiShared.planningData.vec_emergency_planned_roll[self.index[0]])-np.array(self.guiShared.planningData.vec_emergency_planned_roll_var[self.index[0]]))
                #self.line_emergency_roll_var2.set_data(np.arange(len(self.guiShared.planningData.vec_emergency_planned_roll[self.index[0]])),np.array(self.guiShared.planningData.vec_emergency_planned_roll[self.index[0]])+np.array(self.guiShared.planningData.vec_emergency_planned_roll_var[self.index[0]]))
                self.guiShared.update_data_flag = False

            with self.guiShared.Lock:
                path = copy.deepcopy( self.guiShared.real_path)

            if len(path.distance) > 0:
                #if self.index[0] == -1:
                #    max_time_ind = len(path.time)
                #else:
                #    max_time_ind = self.index[0]
                max_time_ind = len(path.time) if self.index[0] == -1 else self.index[0]
                self.line1.set_data(path.time[:max_time_ind],self.guiShared.roll[:max_time_ind])

                update_velocities(self.lines,path,self.guiShared.planningData.vec_emergency_action,self.index[0])
                self.canvas.draw()
                self.guiShared.update_data_flag = False
        
        if self.guiShared.update_episodes_flag:
            c = []
            for num in self.guiShared.episodes_fails:
                if num == 1:#failed
                    c.append('red')
                elif num == 2:#emergency
                    c.append('blue')
                else:#ok
                    c.append('green')
            self.ax_episodes.clear()
            self.ax_episodes.plot([0,len(self.guiShared.episodes_data)],[1,1],color = 'black')
            self.ax_episodes.scatter(list(range(len(self.guiShared.episodes_data))),self.guiShared.episodes_data,marker='o',color = c)
            self.guiShared.update_episodes_flag = False

        self.root.after(50,self.plot_data)

class Q_plot():
    def __init__(self,root,spinBox,guiShared,index):
        self.guiShared = guiShared
        self.root = root
        self.spinBox = spinBox
        self.index = index
        self.last_index = -1
        self.figure,self.ax  = plt.subplots(1, 1,figsize=(2,2))
        #self.ax.set_ylim(-1.0,1.0)
        #self.ax.set_xlim(-1.0,1.0)

        self.canvas = FigureCanvasTkAgg(self.figure, master= root)
        self.canvas.get_tk_widget().grid(row = 1,column = 1,columnspan = 2)

        self.plot_data()

    def plot_data(self):
        print_debug_data = False
        if self.index[0] >=0:#update data if spin box changed.
            self.index[0] = min(int(self.spinBox.get()),len(self.guiShared.roll)-1)
            if self.index[0]!=self.last_index:
                self.guiShared.update_data_flag = True
                print_debug_data = True
                self.last_index = self.index[0]

        if self.guiShared.update_data_flag:
            with self.guiShared.Lock:
                vec_Q = copy.deepcopy( self.guiShared.planningData.vec_Q)
                action_vec = copy.deepcopy( self.guiShared.planningData.action_vec)
                action_noise_vec = copy.deepcopy( self.guiShared.planningData.action_noise_vec)
            if len(vec_Q)>0:
                if len(vec_Q[self.index[0]])>0:
                    #print(self.guiShared.planningData.vec_Q[self.index[0]])
                    #arr = np.arange(100).reshape((10,10))
                    if print_debug_data:
                        for row in vec_Q[self.index[0]]:
                            for item in row:
                                print(item,end = "    ")
                            print()
                    self.ax.clear()
                    #self.ax.set_ylim(-1.0,1.0)
                    #self.ax.set_xlim(-1.0,1.0)
                    im = self.ax.imshow(vec_Q[self.index[0]],cmap = 'plasma')# "YlGn"
                   # im = self.ax.imshow(arr,cmap = 'plasma')# "YlGn"
                    if len(action_vec[self.index[0]])==2:
                        self.ax.scatter([-action_vec[self.index[0]][1]*5+5],[-action_vec[self.index[0]][0]*5+5],color = 'black')
                        self.ax.scatter([-action_noise_vec[self.index[0]][1]*5+5],[-action_noise_vec[self.index[0]][0]*5+5],color = 'white')
                    else:
                        self.ax.scatter([0],[-action_vec[self.index[0]][0]*5+5],color = 'black')
                        self.ax.scatter([0],[-action_noise_vec[self.index[0]][0]*5+5],color = 'white')
                    #self.ax.set_xticks(np.arange(len(self.guiShared.planningData.vec_Q[self.index[0]])))
                    #self.ax.set_yticks(np.arange(len(self.guiShared.planningData.vec_Q[self.index[0]])))
                    ### ... and label them with the respective list entries
                    #self.ax.set_xticklabels(np.arange(len(self.guiShared.planningData.vec_Q[self.index[0]])))
                    #self.ax.set_yticklabels(np.arange(len(self.guiShared.planningData.vec_Q[self.index[0]])),)
                    #self.figure.colorbar(im, ax=self.ax)
                    self.ax.set_title('Q')
                    self.canvas.draw()
                    self.guiShared.update_episodes_flag = False
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
        self.canvas.grid(row = 1,rowspan = 5)
        self.vehicle_pos = np.array([self.canvas_width*0.5,self.canvas_height*0.85])

        self.draw()


        
    def draw(self):
        if self.index[0] >=0:
            self.index[0] = min(int(self.spinBox.get()),len(self.guiShared.planningData.vec_planned_roll)-1)
        self.canvas.delete("all")
        if self.guiShared.steer is not None:
            self.draw_vehicle(self.guiShared.steer)
        
        if len(self.guiShared.planningData.vec_path)>self.index[0]+1:
            if len(self.guiShared.planningData.vec_path[self.index[0]].position)>1:
                self.draw_path(self.guiShared.planningData.vec_path[self.index[0]].position,color = "red")
        if len(self.guiShared.planningData.vec_predicded_path) > self.index[0]+1:
            if len(self.guiShared.planningData.vec_predicded_path[self.index[0]])>1:
                self.draw_path(self.guiShared.planningData.vec_predicded_path[self.index[0]],color = "green")
        if len(self.guiShared.planningData.vec_emergency_predicded_path) > self.index[0]+1:
            if len(self.guiShared.planningData.vec_emergency_predicded_path[self.index[0]])>1:
                self.draw_path(self.guiShared.planningData.vec_emergency_predicded_path[self.index[0]],color = "blue")
        if len(self.guiShared.planningData.target_point)>self.index[0]+1:
            if len(self.guiShared.planningData.target_point[self.index[0]])>1:
                self.draw_point(self.guiShared.planningData.target_point[self.index[0]])
        #draw target points:
        #if len(self.guiShared.planningData.vec_target_points) > self.index[0]+1:
        #    for target_point in self.guiShared.planningData.vec_target_points[self.index[0]]:
        #        self.draw_target_point(target_point)
        self.root.after(50,self.draw)
        #self.draw_vehicle(0.5)
        #test_path = [[0,0],[1,1],[2,4],[5,7],[7,7]]
        #self.draw_path(test_path)
    def draw_point(self,point):
        r = 0.5*self.scale
        x,y = self.vehicle_pos[0]+point[0]*self.scale ,self.vehicle_pos[1]-point[1]*self.scale 
        self.canvas.create_oval(x-r, y-r, x+r, y+r,fill = "black")

    def draw_target_point(self,targetPoint):
        r = 0.5*self.scale
        x,y = self.vehicle_pos[0]+targetPoint.abs_pos[0]*self.scale ,self.vehicle_pos[1]-targetPoint.abs_pos[1]*self.scale 
        self.canvas.create_oval(x-r, y-r, x+r, y+r,fill = "black")

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
        #print("color:",color,"path:",relative_path)
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
        #self.root.geometry('%dx%d+%d+%d' % (screenwidth*0.3,screenheight*0.9,0,0))#(w, h, x, y)
        self.root.wm_title("Gui")

        
        self.spinBox = tk.Spinbox(self.root, from_=0, to= 120)
        self.spinBox.grid(row = 5, column=1,columnspan = 2)
        data_plots(self.root,self.spinBox,self.guiShared,self.index)
        draw_state(self.root,self.spinBox,self.guiShared,self.index)
        if self.guiShared.env_mode != 'model_based':
            Q_plot(self.root,self.spinBox,self.guiShared,self.index)

        #tk.Button(self.root, text="Quit", command=self.quit).pack()
        tk.Button(self.root, text="Quit", command=self.quit).grid(row = 2,column =1)

        self.pause_btn_text = tk.StringVar()
        tk.Button(self.root,  textvariable = self.pause_btn_text, command=self.pause_after_episode).grid(row = 3, column=1)
        self.pause_btn_text.set("Pause")

        self.evaluate_btn_text = tk.StringVar()
        tk.Button(self.root,  textvariable = self.evaluate_btn_text, command=self.evaluate).grid(row = 3, column=2)
        self.evaluate_btn_text.set("Evaluate")

        self.replay_btn_text = tk.StringVar()
        tk.Button(self.root,  textvariable = self.replay_btn_text, command=self.replay_mode).grid(row = 4, column=1)
        self.replay_btn_text.set("Set Replay Mode")

        self.root.mainloop()

    def evaluate(self):
        print("pause at the end of the episode")
        if self.guiShared.evaluate:
            self.guiShared.evaluate = False
            self.evaluate_btn_text.set("Evaluate")
        else:
            self.guiShared.evaluate = True
            self.evaluate_btn_text.set("Cancel evaluate")
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

