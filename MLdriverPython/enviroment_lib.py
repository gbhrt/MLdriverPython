
def choose_points(local_path,number,distance_between_points):#return points from the local path, choose a point every "skip" points
    index = 0
    last_index = 0
    points = []
    points.append(local_path.velocity_limit[index])
    for _ in range(number-1):
        while local_path.distance[index] - local_path.distance[last_index] < distance_between_points: #search the next point 
            if index >= len(local_path.distance)-1:#at the end of the path - break, cause doublication of the last velocity limit
                break
            index += 1
        points.append(local_path.velocity_limit[index])
        last_index = index
    return points
def choose_curvature_points(local_path,number,distance_between_points):#return points from the local path, choose a point every "skip" points
    index = 0
    last_index = 0
    points = []
    points.append(local_path.curvature[index])
    for _ in range(number-1):
        while local_path.distance[index] - local_path.distance[last_index] < distance_between_points: #search the next point 
            if index >= len(local_path.distance)-1:#at the end of the path - break, cause doublication of the last velocity limit
                break
            index += 1
        points.append(local_path.curvature[index])
        last_index = index
    return points
def choose_points2(local_path,number,distance_between_points):#return points from the local path, choose a point every "skip" points
    index = 0
    last_index = 0
    end_flag = False
    points = []
    points.append(local_path.position[index][0])
    points.append(local_path.position[index][1])
    while len(points) < number*2 and end_flag == False:
        while local_path.distance[index] - local_path.distance[last_index] < distance_between_points: #search the next point 
            if index >= len(local_path.distance)-1:#at the end of the path - break, cause doublication of the last velocity limit
                end_flag = True
                break
            
            index += 1
        points.append(local_path.position[index][0])
        points.append(local_path.position[index][1])
        last_index = index
    if len(local_path.position) >=2:
        dis = lib.dist(local_path.position[-1][0],local_path.position[-1][1],local_path.position[-2][0],local_path.position[-2][1])
    else:
        dis = 0.05
    ang = local_path.angle[-1][1]
    while len(points) < number * 2:
        points.append(local_path.position[-1][0] + dis* np.sin(ang))
        points.append(local_path.position[-1][1] + dis* np.cos(ang))

    return points

def get_state(pl = None,local_path = None,num_points = 1,distance_between = 1.0,max_velocity = 30.0,max_curvature = 0.12):
    #velocity limit state:
    #points = choose_points(local_path,points,distance_between)

    #path state:
    #points = choose_points2(local_path,points,distance_between)

    #curvature state:
    points = choose_curvature_points(local_path,num_points,distance_between)
    points = [pnt/max_curvature for pnt in points]
    vel = max(pl.simulator.vehicle.velocity/max_velocity,0)
    state = [vel] +  points
    #state = lib.normalize(state,0,30)




    #path_ang = local_path.angle[0][1]
    #state.append(path_ang)
    #point1 = math.copysign(dist(local_path.position[0][0],local_path.position[0][1],0,0),local_path.position[0][0])#distance from path
    #state.append(point1)

    #for i in range (points):
    #    state.append(local_path.position[i][0])#x
    #    state.append(local_path.position[i][1])#y
    #state.append(local_path.position[target_index][0])#x
    #state.append(local_path.position[target_index][1])#y

    return state

def get_reward(velocity,max_vel,mode):  
    reward = velocity/max_vel 
    if velocity <= 0.0:
        reward = -1.0
    if mode == 'kipp':
        reward = -5.0

    
    #if state[1] < 0.01: # finished the path
    #    reward =(max_velocity - state[0])
    #else:
    #    reward =  0#state[0]*0.01


    #velocity_limit = state[1]
    #acc = state[0] - last_state[0]#acceleration
    #if state[0] < velocity_limit:
    #    if last_state[0] > velocity_limit:
    #        reward = -acc 
    #    else:
    #        reward = acc 
    #else:

    #    reward = -acc
    #acc = state[0] - last_state[0]#acceleration
    #if state[0] < 0:
    #    if last_state[0] > velocity_limit:
    #        reward = -acc 
    #    else:
    #        reward = acc 
    #else:
    #    reward = -acc
    #if velocity < velocity_limit: 
    #    reward = velocity
    #    #if velocity < 1e-5:
    #    #    reward = - 2
    #else: 
    #    reward = -10.*(velocity - velocity_limit)

    #if state[0] < 0: 
    #    reward = velocity_limit + state[0]
    #    if reward < 0.01:
    #        reward = - 2
    #else: 
    #    reward = -10*state[0]

    return reward
