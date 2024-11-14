import pygame
from pygame.locals import *
import gymnasium as gym
from gymnasium.envs.registration import register
from gymnasium.utils.env_checker import check_env
import numpy as np
from gymnasium import spaces
from run import GameController
from constants import *
from DQN_model import DQN_model
from enviromentConstants import *
from stable_baselines3 import DQN
from stable_baselines3.dqn import MultiInputPolicy
import os


GHOST_MODES = {SCATTER: 0, CHASE: 0, FREIGHT: 1, SPAWN: 2}


if "pacman-v0" not in gym.envs.registry:
    register(id="pacman-v0", entry_point="pacman_env:PacmanEnv", max_episode_steps=1000)


class PacmanEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self, render_mode=None):
        self.game = GameController(rlTraining=True)

        self.observation_space = spaces.Dict(
            {
                "walls_position": spaces.Box(
                    low = np.array([0,0,0,0]), high = np.array([1,1,1,1]), dtype=np.bool_
                ),
                "best_direction": spaces.Box(
                    low = np.array([-2]), high = np.array([2]) , dtype=np.int_
                ),
                "ghosts_position": spaces.Box(
                    low = np.array([0,0,0,0]), high = np.array([1,1,1,1]), dtype=np.bool_
                ),
                "trapped": spaces.Box(
                    low = np.array([False]), high = np.array([True]) , dtype=np.bool_
                )
            }
        )
        self.action_space = spaces.Discrete(5, start=0)

        self._walls_position = np.array([0, 0 , 0 , 0], dtype=np.bool_)
        self._best_direction = np.array([0] , dtype=np.int_)
        self._ghosts_position = np.array([0, 0 , 0 , 0], dtype=np.bool_)
        self._trapped = np.array([True], dtype=np.bool_)


        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        if self.render_mode == "human":
            self.window = self.game.screen
            self.clock = self.game.clock

    def _getobs(self):
        self._action_to_direction = self.game.pacman.directions
        self._walls_position , self._ghosts_position , ghosts_rewards = self.game.observation

        max_ghost_avoidance = max(ghosts_rewards[direction][1] for direction in ghosts_rewards)
        filtered_dir = {direction : val for direction , val in ghosts_rewards.items() if val[1] == max_ghost_avoidance}
        best_dir = max(filtered_dir, key=lambda direction: filtered_dir[direction][0])
        self._best_direction = np.array([best_dir])

        trapped = not(any(val[1] in [100,101] for val in ghosts_rewards.values()))
        self._trapped = np.array([trapped])

        """
        print({
            "walls_position": self._walls_position,
            "best_direction": self._best_direction,
            "ghosts_position" :  self._ghosts_position,
            "trapped": self._trapped
        })
        """
        return {
            "walls_position": self._walls_position,
            "best_direction": self._best_direction,
            "ghosts_position" :  self._ghosts_position,
            "trapped": self._trapped
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.game.restartGame()

        observation = self._getobs()
        info = {}
        return observation, info

    def step(self, action):
        action -= 2
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
        return observation, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "human":
            self.game.render()

    def close(self):
        if self.window is not None:
            pygame.event.post(pygame.event.Event(QUIT))

def constant_lr(_):
    return 1e-3  # Fixed learning rate of 0.001


if __name__ == "__main__":
    env_not_render = gym.make("pacman-v0", max_episode_steps = 500)
    env_render = gym.make("pacman-v0", max_episode_steps = 500 , render_mode = "human")
    
    model_path = "./models/simple_DQN_baseline_model_40,000,000_steps"
    log_path = "./logs/fit"

    
    if not os.path.exists(log_path):
        os.makedirs(log_path)


    if not os.path.exists(model_path):  
        os.makedirs(model_path)
        env = env_not_render
        obs , _ = env.reset()
        
        ##policy netwok for the model [64,64,32]
        model = DQN("MultiInputPolicy" , env ,  verbose = 1 , tensorboard_log = log_path)

        time_steps = 4000_000
        for i in range (10):
            model.learn(total_timesteps = time_steps , progress_bar=True , reset_num_timesteps = False , tb_log_name = "simple_DQN_baseline_model_40,000,000_steps_2")
            model.save(f"{model_path}/{(i+1)*time_steps}")



    elif os.path.exists(model_path):
        env = env_render
        obs , _ = env.reset()
        model_final_path = f"{model_path}/800000.zip"
        model = DQN.load(model_final_path , env = env)

        episodes = 10
        for ep in range(episodes):
            done = False
            while not done:
                action , next_state = model.predict(obs)
                obs, reward, terminated, truncated, info = env.step(int(action))
                done = terminated
        env.close()