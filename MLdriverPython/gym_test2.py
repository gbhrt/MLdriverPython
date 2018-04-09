import gym
import pybullet_envs


env = gym.make("HalfCheetahBulletEnv-v0")
env.render(mode="human")
env.reset()