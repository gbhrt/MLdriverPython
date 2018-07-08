import math
import pyglet
import numpy as np
from pyglet.gl import *
import threading
import time



# get all the points in a circle centered at 0.
def PointsInCircum(r, n=25, pi=3.14):
    return [(math.cos(2*pi/n*x)*r,math.sin(2*pi/n*x)*r) for x in range(0,n+1)]
pts = np.array(PointsInCircum(20))

# function that increments to the next
# point along a circle
frame = 0
def update_frame(x, y):
    global frame
    if frame == None or frame == pts.shape[0]-1:
        frame = 0
    else:
        frame += 1
    return

def draw_path(position):
    #print(path.position[:5])
    glBegin(GL_LINE_STRIP)
    for pos in position:
        glVertex3f(pos[0],pos[1],0)
    glEnd()
    glPointSize(2.0)
    glBegin(GL_POINTS)
    for pos in position:
        glVertex3f(pos[0],pos[1],0)
    glEnd()


def draw_vehicle(steer):
    width = 2.08
    length = 4.0
    glBegin(GL_QUADS)
    glVertex3f(-width/2,length/2,0)
    glVertex3f(width/2,length/2,0)
    glVertex3f(width/2,-length/2,0)
    glVertex3f(-width/2,-length/2,0)
    glEnd()
    return

def draw_win(win,shared):
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluOrtho2D(-20,20,-10,30)


    #glClearColor(1.0,1.0,1.0,1.0);

    #glPushMatrix()
    glColor3f(0,0,1)
    draw_vehicle(shared.steer)
    glColor3f(0,1,0)
    if shared.state is not None:
        draw_path(shared.state['path'].position)
    glColor3f(1,0,0)
    if shared.predicded_path is not None:
        draw_path(shared.predicded_path)
    #glPopMatrix()

#if time.time() - start_time > 5:
#    shared.exit_program = True

def update():
    return

def start_gui(shared):
    win = pyglet.window.Window()
    start_time = time.time()

    glClear(GL_COLOR_BUFFER_BIT)
    glViewport(0,0,win.width,win.height)
    glMatrixMode(GL_PROJECTION)

    @win.event
    def on_draw():
        # clear the screen
        glClear(GL_COLOR_BUFFER_BIT)
        # draw the next line
        # in the circle animation
        # circle centered at 100,100,0 = x,y,z

        

        #glBegin(GL_LINES)
        #glVertex3f(0,0,0)
        #glVertex3f(pts[frame][1],pts[frame][0],0)
        #glEnd()
        draw_win(win,shared)

        #test(win)
        

            

    # every 1/10 th get the next frame
    pyglet.clock.schedule(update_frame, 1/10.0)
    pyglet.app.run()


