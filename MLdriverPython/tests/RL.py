import gym
from getchClass import getch
import numpy as np
import random
import time


def play():
    env = gym.make('Pong-v0')
    env.reset()
    env.render()
    move = getch()
    while move.isdigit():
        dig_move = int(move)
        if(dig_move < env.action_space.n):
            env.step(dig_move)
            env.render()
        move = getch()
        print(move)
    return

def Q_learn_frozen_lake():
    env = gym.make('FrozenLake-v0')
    env.reset()
    env.render()

    epsilon = 0.1
    gamma = 0.95#discount factor
    alpha = 0.1#learnimg rate
    Q = np.zeros ([env.observation_space.n,env.action_space.n])
    for i in range(1000): #number of games
       s = env.reset()
       env.render()
       done = False
       while not done: #until game over, or max number of steps
          if random.random() < epsilon:
              a = env.action_space.sample()
          else:
              a = np.argmax(Q[s,:])
          s_n, r, done, _ = env.step(a)
          if r >0:
            print("s_n: ",s_n," r: :", r," s: ",s)
            if getch() == 'x':
                break
          env.render()
          time.sleep(0.1)
          
          Q[s,a] = Q[s,a] + alpha*(r + gamma * np.max(Q[s_n,:]) - Q[s,a])
          s = s_n
    return

def random_player():
    env = gym.make('Pong-v0')
    env.reset()
    for _ in range(1000):
        env.render()
        env.step(env.action_space.sample()) # take a random action
        time.sleep(0.03)


#random_player()
#play()
Q_learn_frozen_lake()