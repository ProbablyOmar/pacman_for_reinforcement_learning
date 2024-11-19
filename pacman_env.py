import pygame
from pygame.locals import *
import gymnasium as gym
from gymnasium.envs.registration import register
from gymnasium.utils.env_checker import check_env
import numpy as np
from gymnasium import spaces
from run import GameController
from constants import *
from tqdm import tqdm
import time
import random
from enviromentConstants import *
import matplotlib.pyplot as plt
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
        self.action_space = spaces.Discrete(5, start=-2)

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

def get_model_obs(dict_state):
    model_state = np.array([])

    for val in dict_state.values():
        model_state = np.append(model_state, val)

    return model_state