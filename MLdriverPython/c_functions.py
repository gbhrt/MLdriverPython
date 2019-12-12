import os
from ctypes import *
#import ctypes
#libc = cdll.msvcrt 
#libc.printf
class cFunctions:
    def __init__(self):
        try:
            self.c_lib = CDLL(os.getcwd()+'/c_lib.dll')#windows
        except:
            self.c_lib = CDLL(os.getcwd()+'/c_lib.so')#linux
        #self.c_lib = CDLL('c_lib.so')
        #func = c_lib['compute_limit_curve']
        #self.c_lib.compute_limit_curve.argtypes = (c_int, POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float))
        #self.c_lib.compute_limit_curve.restype = c_int
        self.c_lib.compute_limit_curve_and_velocity.argtypes = (c_int, POINTER(c_float),POINTER(c_float),POINTER(c_float),c_float,c_float,\
            c_float,c_float,c_float,c_float,c_float,c_float,\
            POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float))
        self.c_lib.compute_limit_curve_and_velocity.restype = c_int

    def comp_limit_curve(self,x,y,z):
        #global c_lib
        num = len(x)
        limit_curve = [0 for _ in range(num)]
        data_type = (c_float * num)
        c_limit_curve =  data_type(*limit_curve)
        c_x = data_type(*x)
        c_y = data_type(*y)  
        c_z = data_type(*z)
        result = self.c_lib.compute_limit_curve(c_int(num),c_x,c_y,c_z,c_limit_curve)#c_int(num), array_type(*limit_curve)
        #if result == 1:
        #    return 1
        return c_limit_curve[:]
    def comp_limit_curve_and_velocity(self,x,y,z,init_vel = 0, final_vel = 0):
        f_max = 4195*5.0# 4195# = 2000 torque.  8864# from 0-100 in 10 sec  4195#*5.0
        mass = 3200
        vel_max = 30
        fc = 1.0
        height = 1.7#1.7#0.94#0.86 all test for the proposal done with 1.7
        width = 2.08

        self.max_acc= f_max/mass


        #global c_lib
        num = len(x)
        limit_curve = [0 for _ in range(num)]
        velocity = [0 for _ in range(num)]
        time_vec = [0 for _ in range(num)]
        acc_vec = [0 for _ in range(num)]
        data_type = (c_float * num)
        c_limit_curve =  data_type(*limit_curve)
        c_velocity =  data_type(*velocity)
        c_time =  data_type(*time_vec)
        c_acc_vec = data_type(*acc_vec)
        c_x = data_type(*x)
        c_y = data_type(*y)  
        c_z = data_type(*z)
        result = self.c_lib.compute_limit_curve_and_velocity(c_int(num),c_x,c_y,c_z,c_float(init_vel),c_float(final_vel),\
            c_float(f_max),c_float(mass),c_float(vel_max),c_float(fc),c_float(height),c_float(width),\
            c_limit_curve,c_velocity,c_time,c_acc_vec)#c_int(num), array_type(*limit_curve)
        #if result == 1:
        #    return 1
        return c_limit_curve[:], c_velocity[:],c_time[:],c_acc_vec[:],result

    #x = [0]*100
    #y = [x/10 for x in range(100)]
    #z = [0]*100
    #print(x)
    #print(y)
    #print(z)
    #print(compute_limit_curve(x,y,z ))
    #def our_function(numbers):
    #    global _sum
    #    num_numbers = len(numbers)
    #    array_type = c_int * num_numbers
    #    result = _sum.our_function(c_int(num_numbers), array_type(*numbers))
    #    return int(result)

    #print(our_function([1,2,3]))


