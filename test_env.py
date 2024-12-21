import gymnasium as gym
from gymnasium.envs.registration import register
from gymnasium.utils.env_checker import check_env
from gymnasium import spaces
from pacman_env import PacmanEnv
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
from constants import *
import os


os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
env = DummyVecEnv([lambda: PacmanEnv(render_mode = "human" , mode = SAFE_MODE , move_mode = DISCRETE_STEPS_MODE, clock_tick = 0 , pacman_lives = 1)])
env = VecFrameStack(env, 1, channels_order='last')
# print("Checking Environment")
# check_env(env.unwrapped)
# print("done checking environment")

obs = env.reset()[0]
done = False
action = 4
while not done:
    randaction = env.action_space.sample()
    env.render()
    obs, reward, terminated, _, _ = env.step(action)
    done = terminated
    print(obs)
    #print(reward)
    if action == 1 and reward == HIT_WALL_PENALITY:
        action = 2
    elif reward == HIT_WALL_PENALITY:
        action = 1