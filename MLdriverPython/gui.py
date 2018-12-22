import math
import pyglet
import numpy as np
from pyglet.gl import *
#import threading
import time
#import ion_graph
import copy

def PointsInCircum(r, n=25, pi=3.14):
    return [(math.cos(2*pi/n*x)*r,math.sin(2*pi/n*x)*r) for x in range(0,n+1)]
pts = np.array(PointsInCircum(20))
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

def draw_wheel():
    w = 0.5
    l = 1
    glBegin(GL_QUADS)
    glVertex3f(-w/2,l/2,0)
    glVertex3f(w/2,l/2,0)
    glVertex3f(w/2,-l/2,0)
    glVertex3f(-w/2,-l/2,0)
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
    #draw_wheel()
    glPushMatrix()
    glTranslatef(2.08/2+0.27,-length/2,0)
    draw_wheel()
    glPopMatrix()

    glPushMatrix()
    glTranslatef(-(2.08/2+0.27),-length/2,0)
    draw_wheel()
    glPopMatrix()

    steer_deg = steer*180/3.141
    glPushMatrix()
    glTranslatef((2.08/2+0.27),length/2,0)
    glRotatef(steer_deg,0,0,1)
    draw_wheel()
    glPopMatrix()

    glPushMatrix()
    glTranslatef(-(2.08/2+0.27),length/2,0)
    glRotatef(steer_deg,0,0,1)
    draw_wheel()
    glPopMatrix()

    return

def draw_win(win,guiShared):
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluOrtho2D(-20,20,-10,30)


    #glClearColor(1.0,1.0,1.0,1.0);

    #glPushMatrix()
    glColor3f(0,0,1)
    if guiShared.steer is not None:
        draw_vehicle(guiShared.steer)
    glColor3f(0,1,0)
    if guiShared.state is not None:
        draw_path(guiShared.state['path'].position)
    glColor3f(1,0,0)
    if guiShared.predicded_path is not None:
        draw_path(guiShared.predicded_path)
    #glPopMatrix()

#if time.time() - start_time > 5:
#    guiShared.exit_program = True

def update():
    return

def start_gui(guiShared):
    #ionGraph = ion_graph.ionGraph()

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
        with guiShared.Lock: 
            draw_win(win,guiShared)
            if guiShared.exit_program:
                pyglet.app.exit()
        #ionGraph.update(guiShared)
        #test(win)
        

         

    # every 1/10 th get the next frame
    pyglet.clock.schedule(update_frame, 1/10.0)
    pyglet.app.run()
    

