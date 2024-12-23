import pygame
from pygame.locals import *
import gymnasium as gym
from gymnasium.envs.registration import register
from gymnasium.utils.env_checker import check_env
import numpy as np
from gymnasium import spaces
from run import GameController
from constants import *
from DQN_model import CustomCNN
from stable_baselines3 import DQN
from stable_baselines3.dqn import MultiInputPolicy
import os
import copy

GHOST_MODES = {SCATTER: 0, CHASE: 0, FREIGHT: 1, SPAWN: 2}


if "pacman-v0" not in gym.envs.registry:
    register(id="pacman-v0", entry_point="pacman_env:PacmanEnv", max_episode_steps=1000)


class PacmanEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self, render_mode=None):
        self.game = GameController(rlTraining=True , mode = SAFE_MODE)
        self.game_score = 0
        self.useless_steps = 0

        self.observation_space = spaces.Box(
                    low = 0, high = 13 , shape = (1 , GAME_ROWS , GAME_COLS) , dtype=np.int_
                )
        
        self.action_space = spaces.Discrete(5, start=0)

        self._maze_map = np.zeros(shape=(GAME_ROWS , GAME_COLS), dtype=np.int_)
        self._last_obs = np.zeros(shape=(GAME_ROWS , GAME_COLS), dtype=np.int_)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        if self.render_mode == "human":
            self.window = self.game.screen
            self.clock = self.game.clock

    def _getobs(self):
        self._maze_map = self.game.observation
        self._maze_map = np.expand_dims(self._maze_map , axis=0)

        return self._maze_map

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.game.restartGame()

        observation = self._getobs()
        info = {}
        return observation, info

    def step(self, action):
        action -= 2
        step_reward = TIME_PENALITY
        while True:
            if self.render_mode == "human":
                self.game.update(
                    agent_direction=action,
                    render=True,
                    clocktick=self.metadata["render_fps"],
                )
            else:
                self.game.update(
                    agent_direction=action,
                    render=False,
                    clocktick=self.metadata["render_fps"],
                )
            
            terminated = self.game.done
            truncated = False
            reward = self.game.RLreward
            observation = self._getobs()
            info = {}

            if reward != TIME_PENALITY:
                step_reward = reward

            if not np.array_equal(observation , self._last_obs): 
                np.copyto(self._last_obs , observation)
                self.game_score += step_reward

                if self.game.mode == SAFE_MODE:
                    if reward == TIME_PENALITY or reward == HIT_WALL_PENALITY:
                        self.useless_steps +=1
                        if self.useless_steps >= MAX_USELESS_STEPS:
                            self.game.done = True
                            terminated = self.game.done
                            self.useless_steps = 0
                    # else:
                    #     self.useless_steps = 0
                return observation, step_reward, terminated, truncated, info 

    def render(self):
        if self.render_mode == "human":
            self.game.render()

    def close(self):
        if self.window is not None:
            pygame.event.post(pygame.event.Event(QUIT))


if __name__ == "__main__":
    env_not_render = gym.make("pacman-v0", max_episode_steps = 10_000)
    env_render = gym.make("pacman-v0", max_episode_steps = 10_000 , render_mode = "human")
    
    model_path = "./models/dqn_baseline_cnn2"
    log_path = "./logs/fit"
   
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    if not os.path.exists(model_path):  
        os.makedirs(model_path)
        env = env_not_render
        obs , _ = env.reset()

        policy_kwargs = dict(
            features_extractor_class=CustomCNN,
            features_extractor_kwargs=dict(features_dim=256),
        )

        model = DQN(
            "CnnPolicy" , 
            env , 
            learning_rate  = 0.0001 , 
            learning_starts  = 2000,
            batch_size= 32,   #32
            gamma = 0.97,
            #train_freq = (1, "episode"),
            gradient_steps = 4,
            target_update_interval=100,
            exploration_fraction=0.99,
            exploration_initial_eps=1,
            exploration_final_eps=0.1,

            policy_kwargs = policy_kwargs , 
            verbose = 1 , 
            tensorboard_log = log_path
        )

        time_steps = 400_000
        for i in range (100):
            model.learn(total_timesteps = time_steps , progress_bar=True , reset_num_timesteps = False , tb_log_name = "./cnn/dqn_baseline_cnn2")
            model.save(f"{model_path}/{(i+1)*time_steps}")

    elif os.path.exists(model_path):
        env = env_render
        obs , _ = env.reset()
        model_final_path = f"{model_path}/400000.zip"
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


# if __name__ == "__main__":
#     os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
#     env = gym.make("pacman-v0", render_mode="human")
#     # print("Checking Environment")
#     # check_env(env.unwrapped)
#     # print("done checking environment")

#     obs = env.reset()[0]
#     done = False
#     action = 4
#     while not done:
#         randaction = env.action_space.sample()
#         env.render()
#         obs, reward, terminated, _, _ = env.step(action)
#         done = terminated 
#         print(obs.shape)
#         # print(reward)
#         if action == 1 and reward == HIT_WALL_PENALITY:
#             action = 2
#         elif reward == HIT_WALL_PENALITY:
#             action = 1




import torch as th
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class CustomCNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        n_input_channels = 1
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 8, kernel_size=2, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=2, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=2, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(
                th.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim), 
            nn.ReLU()
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))