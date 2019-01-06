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
    #lines.append(fig.plot(np.array(path.distance)[:max_time_ind],np.array(path.analytic_velocity_limit)[:max_time_ind],label = "velocity limit")[0])
    #lines.append(fig.plot(np.array(path.distance)[:max_time_ind],np.array(path.analytic_velocity)[:max_time_ind],label = "analytic velocity")[0])
    #lines.append(fig.plot(path.distance,path.velocity,'o', label = "vehicle velocity")[0])
    x,y = [],[]
    lines.append(fig.plot(x,y,label = "velocity limit")[0])
    lines.append(fig.plot(x,y,label = "analytic velocity")[0])
    lines.append(fig.plot(x,y,label = "vehicle velocity")[0])
    plt.legend()
    return lines

def update_velocities(lines,path):
    max_time_ind = len(path.time)
    lines[0].set_data(np.array(path.distance)[:max_time_ind],np.array(path.analytic_velocity_limit)[:max_time_ind])
    
    lines[1].set_data(np.array(path.distance)[:max_time_ind],np.array(path.analytic_velocity)[:max_time_ind])
    lines[2].set_data(path.distance,path.velocity)

class data_plots():
    def __init__(self,root,guiShared,dataManager):
        self.guiShared = guiShared
        self.dataManager = dataManager
        self.root = root


        #self.figure1 = plt.Figure(figsize=(6,5))#, dpi=100)
        #self.ax1 = self.figure1.add_subplot(111)
        #self.ax2 = self.figure1.add_subplot(121)
        self.figure1, (self.ax1,self.ax2) = plt.subplots(2, 1,figsize=(5,4))
        #self.dataManager.roll = [0,1]
        self.line1 = self.ax1.plot(np.array(self.dataManager.roll))[0]
        self.lines = plot_velocities(self.ax2)#,self.dataManager)

        #self.figure2, self.ax1 = plt.subplots(1, 1,figsize=(5,4))


        self.canvas = FigureCanvasTkAgg(self.figure1, master= root)
        #self.canvas.get_tk_widget().pack()#side=tk.LEFT, fill=tk.BOTH)
        self.canvas.get_tk_widget().grid(row = 0,column = 0, columnspan=2)
        self.plot_data()

    def plot_data(self):
        path = self.dataManager.real_path
        #self.dataManager.roll = [1,3,2]
        #plt.cla()
        #self.ax1.clear()
        #self.ax1.plot(self.dataManager.roll)
        #print("roll:",self.dataManager.roll)
        #print("len roll:",len(self.dataManager.roll))
        #print("dis:",path.distance)

        if len(path.distance) > 0:
            self.ax1.set_xlim(0,path.distance[len(self.dataManager.roll)-1])
            self.ax1.set_ylim(-0.1,0.1)
            self.line1.set_data(path.distance[:len(self.dataManager.roll)],np.array(self.dataManager.roll))
            self.ax2.set_xlim(0,path.distance[len(self.dataManager.roll)-1])
            self.ax2.set_ylim(0,25)
        #plot_velocities(self.ax2,self.dataManager)
            update_velocities(self.lines,path)
        self.canvas.draw()

        self.root.after(1,self.plot_data)

#class cordinates:
#    def __init__(self):
#        self.origin = np.array([0.0,0.0])
#    def translated(self,cords):
#        return list(np.array(cords) + self.origin)
#    def translate(self,delta_cords):
#        self.origin += np.array(delta_cords)

#        return
class draw_state:
    def __init__(self,root,guiShared,dataManager):
        self.guiShared = guiShared
        self.dataManager = dataManager
        self.root = root
        
        
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
        self.canvas.delete("all")
        if self.guiShared.steer is not None:
            self.draw_vehicle(self.guiShared.steer)
        if self.guiShared.state is not None:
            if len(self.guiShared.state['path'].position)>1:
                self.draw_path(self.guiShared.state['path'].position,color = "red")
            if len(self.guiShared.predicded_path)>1:
                self.draw_path(self.guiShared.predicded_path,color = "green")

        self.root.after(1,self.draw)
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
    def __init__(self,guiShared,dataManager):
        self.guiShared = guiShared
        self.dataManager = dataManager
        self.root= tk.Tk() 
        screenwidth = self.root.winfo_screenwidth()
        screenheight = self.root.winfo_screenheight()
        self.root.geometry('%dx%d+%d+%d' % (screenwidth*0.4,screenheight*0.9,0,0))#(w, h, x, y)
        self.root.wm_title("Gui")
        data_plots(self.root,guiShared,dataManager)
        draw_state(self.root,guiShared,dataManager)

        #tk.Button(self.root, text="Quit", command=self.quit).pack()
        tk.Button(self.root, text="Quit", command=self.quit).grid(row = 1,column =1)

        self.btn_text = tk.StringVar()
        tk.Button(self.root,  textvariable = self.btn_text, command=self.pause_after_episode).grid(row = 2, column=1)
        self.btn_text.set("Pause")
        self.root.mainloop()

    def quit(self):
        self.guiShared.request_exit = True
        while not self.guiShared.exit:#wait for finish the program
            time.sleep(0.1)
        self.root.destroy()
    def pause_after_episode(self):
        if self.guiShared.pause_after_episode_flag:
            self.guiShared.pause_after_episode_flag = False
            self.btn_text.set("Pause")
        else:
            self.guiShared.pause_after_episode_flag = True
            self.btn_text.set("Continue")
            
        

        

        #self.draw_graph()
        #self.root.mainloop()
        return
    

    #def draw_graph(self):
       
    #    self.bar1.draw()
    #    print("roll:--------------------------------",self.dataManager.roll)
        
    #    self.root.after(1000,self.draw_graph())

