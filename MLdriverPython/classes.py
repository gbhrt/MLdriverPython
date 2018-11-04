import math
import json
import numpy as np
import os
import library as lib
import _thread


class Vehicle:
    def __init__(self):
        self.position=[0,0,0]
        self.backPosition = [0,0,0]
        self.angle = [0,0,0]
        self.steering = 0
        self.velocity = 0
        self.tan_velocity = 0
        self.tan_acceleration = 0
        self.wheels_vel = [0,0,0,0]
class Path:
    def __init__(self):
        self.position = []#
        self.backPosition = []
        self.angle = []
        self.curvature = []
        self.velocity = [] #real velocity for a real path and planed velocity for a planned path
        self.steering = []
        self.distance = []
        self.time = []
        self.max_velocity = []#maximum velocity at each time - maximum allowed velocity
        self.analytic_velocity_limit = []#velocity limit at each point (from analitic compute)
        self.analytic_velocity = []
        self.analytic_acceleration = []
        self.analytic_time = []
        self.seed = None
  
    def dist(self,x1,y1,x2,y2):
        tmp = (x2-x1)**2 + (y2- y1)**2
        if tmp > 0:
            return math.sqrt(tmp)
        else:
            return 0.
    def comp_curvature(self):
       
        for i in range ( len(self.position)-2):#start from 0 up to end - 2
            pnt1 = np.array([self.position[i][0],self.position[i][1],self.position[i][2]])
            pnt2 = np.array([self.position[i+1][0],self.position[i+1][1],self.position[i+1][2]])
            pnt3 = np.array([self.position[i+2][0],self.position[i+2][1],self.position[i+2][2]])
            self.curvature.append(lib.comp_curvature(pnt1,pnt2,pnt3))
        self.curvature.append(self.curvature[-1])
        self.curvature.append(self.curvature[-1])
        #self.curvature = [abs(curv) for curv in self.curvature]

    def comp_distance(self):
        self.distance = []
        
        self.distance.append(0.)
        for i in range (1,len(self.position)):#start from 2
            self.distance.append(self.distance[i-1] + self.dist(self.position[i][0],self.position[i][1],self.position[i-1][0],self.position[i-1][1]))
        return
    def comp_angle(self):
        self.angle =[]
        for i in range (0,len(self.position)-1):#start from 0
           self.angle.append([0,-(math.atan2(self.position[i+1][1] - self.position[i][1],self.position[i+1][0] - self.position[i][0]) - math.pi/2),0])
        self.angle.append(self.angle[-1])
    def set_velocity(self, vel):
        self.velocity = []
        for i in range (len(self.position)):
            self.velocity.append(vel)
        return


class PathManager:#
    def __init__(self):
        self.random_count = 0
        self.max_count = 30
        return
    def read_path_data(self,file_name):#, x,y,steer_ang
        path = Path()
        try:
            with open(file_name, 'r') as f:
                data = f.readlines()
                data = [x.strip().split() for x in data]

                results = []
                for x in data:
                    results.append(list(map(float, x)))
                    pos = [float(x[0]),float(x[1]),float(x[2])]
                    path.position.append(pos)
                    ang = [0,float(x[3]),0]
                    path.angle.append(ang)
                    path.analytic_velocity_limit.append(float(x[4]))
                    path.steering.append(float(x[5]))
                #self.desired_path = path
        except ValueError:
            print("cannot read file",file_name,"ValueError: ",ValueError)
        return path  

    def save_path(self,path,file_name):
        with open(file_name, 'w') as f:
            #for i in range (len(path.position)):
            #    f.write("%s \t %s\t %s\t %s\t %s\n" % (path.position[i][0],path.position[i][1],path.angle[i][1],path.velocity[i],path.steering[i]))
            #f.write("\n")
            json.dump([path.position,path.angle,path.distance,path.velocity,path.analytic_velocity_limit],f)
        return
    def read_path(self,file_name):
        path = Path()
        try:
            with open(file_name, 'r') as f:#append data to the file
                #for i in range (len(path.position)):
                #    f.write("%s \t %s\t %s\t %s\t %s\n" % (path.position[i][0],path.position[i][1],path.angle[i][1],path.velocity[i],path.steering[i]))
                #f.write("\n")
                [path.position,path.angle,path.distance,path.velocity,path.analytic_velocity_limit] = json.load(f)
        except:
            print("cannot read file: ",file_name)
            return None
        return path
    def convert_to_json(self,in_file_name,out_file_name):
        path = self.read_path_data(in_file_name)
        self.save_path(path,out_file_name)
        print("done")
        return
    def copy_path(self,path,start, num_of_points = None):#return path from start to end
        cpath = Path()
        if num_of_points == None:
            end = len(path.position)
        else:
            end = np.clip(start + num_of_points,0,len(path.position))
        if len(path.position) >= end: cpath.position =  path.position[start:end]
        if len(path.angle) >= end: cpath.angle =  path.angle[start:end]
        if len(path.curvature) >= end: cpath.curvature =  path.curvature[start:end]
        if len(path.analytic_velocity_limit) >= end: cpath.analytic_velocity_limit =  path.analytic_velocity_limit[start:end]
        if len(path.analytic_velocity) >= end: cpath.analytic_velocity =  path.analytic_velocity[start:end]

        #for i in range(start,end):
        #    cpath.position.append(path.position[i])
        #    cpath.angle.append(path.angle[i])
        #    cpath.analytic_velocity_limit.append(path.analytic_velocity_limit[i])
        return cpath

    def split_path(self,input_path_name,num_points,output_name):#input file name of a path, split to paths in num_points lenght. save to output_name_i
        in_path = self.read_path(input_path_name)
        location = os.getcwd()
        paths_count = 0
        i = 0
        while i < len(in_path.position):
            out_path = self.copy_path(in_path,i,num_points)
            i+=num_points
            paths_count+=1
            #name = output_name#'splited_files\straight_path_limit2'#input_path_name#
            name = output_name + str(paths_count)+ '.txt'

            self.save_path(out_path,name)#out_name)
        return
    def get_next_random_path(self):
        def read():
            self.random_count+=1
            return self.random_count,self.read_path('splited_files\\random2\\path_'+ str(self.random_count) +'.txt')
        _,path = read()
        if path == None:
            self.random_count = 1
            _,path = read()
            if path == None:
                return self.random_count,None
        return self.random_count,path
    def create_const_curve_path(self):
        dis = 0.05
        R = 8.0
        num_pnts = int(2*math.pi*R/dis * 10)
        
        path = Path()
        
        dang = dis/R
        ang=0
        for i in range(num_pnts):
            ang += dang
            x = R*math.cos(ang)
            y = R*math.sin(ang)
            path.position.append([x,y,0])    
        return path


