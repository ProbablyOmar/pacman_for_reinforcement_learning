import gymnasium as gym
from gymnasium.envs.registration import register
from gymnasium.utils.env_checker import check_env
from gymnasium import spaces
from pacman_env import PacmanEnv
from DQN_model import CustomCNN , CustomCNN_eat_pellets2 , CustomCNN_eat_pellets3
from stable_baselines3 import DQN
from stable_baselines3.dqn import MultiInputPolicy
from stable_baselines3.common.vec_env import DummyVecEnv
from constants import *
import os

env_not_render = DummyVecEnv([lambda: PacmanEnv(mode = SAFE_MODE , move_mode = DISCRETE_STEPS_MODE, clock_tick = 0 , pacman_lives = 1)])
env_render = DummyVecEnv([lambda: PacmanEnv(render_mode = "human" , mode = SAFE_MODE , move_mode = DISCRETE_STEPS_MODE, clock_tick = 0 , pacman_lives = 1)])


model_load_path = "./models/dqn_baseline_cnn3_eat_pellets_3"   #with max_useless_steps = 1000
model_path = "./models/dqn_baseline_cnn3_eat_pellets_4"

log_path = "./logs/fit"

if not os.path.exists(log_path):
    os.makedirs(log_path)

if not os.path.exists(model_path):  
    os.makedirs(model_path) 
    env = env_not_render
    obs , _ = env.reset()

    model_final_path = f"{model_load_path}/400000"

    model = DQN.load(model_final_path , env = env) 
    model.features_extractor_class = CustomCNN_eat_pellets3
    model.features_extractor_kwargs = dict(features_dim=1024),
    model.learning_rate =  0.00005
    model.exploration_fraction = 1,
    model.exploration_initial_eps=1,
    model.exploration_final_eps=0.5,  ## changed in pellets_4

    time_steps = 400_000
    for i in range (100):
        model.learn(total_timesteps = time_steps , progress_bar=True , reset_num_timesteps = False , tb_log_name = "./cnn/dqn_baseline_cnn3_eat_pellets_4")
        model.save(f"{model_path}/{(i+1)*time_steps}") 

elif os.path.exists(model_path):
    env = env_render
    obs , _ = env.reset()
    model_final_path = f"./{model_path}/1600000.zip"
    model = DQN.load(model_final_path , env = env)

    episodes = 10
    for ep in range(episodes):
        done = False
        while not done:
            action , next_state = model.predict(obs)
            obs, reward, terminated, truncated, info = env.step(int(action))
            print(env.game_score)
            done = terminated
    env.close()