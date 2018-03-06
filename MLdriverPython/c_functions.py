import os
from ctypes import *
#import ctypes
#libc = cdll.msvcrt 
#libc.printf
class cFunctions:
    def __init__(self):
        self.c_lib = CDLL('c_lib.dll')
        #func = c_lib['compute_limit_curve']
        self.c_lib.compute_limit_curve.argtypes = (c_int, POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float))
        self.c_lib.compute_limit_curve.restype = c_int

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


