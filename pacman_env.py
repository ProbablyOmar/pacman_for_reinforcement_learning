import pygame
from pygame.locals import *
import gymnasium as gym
from gymnasium.envs.registration import register
from gymnasium.utils.env_checker import check_env
import numpy as np
from gymnasium import spaces
from run import GameController
from constants import *

NOREWARD = 0
PELLETREWARD = 1
POWERPELLETREWARD = 2
FRUITREWARD = 3

if "pacman-v0" not in gym.envs.registry:
    register(id="pacman-v0", entry_point="pacman_env:PacmanEnv", max_episode_steps=1000)


class PacmanEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self, render_mode=None):
        self.game = GameController(rlTraining=True)

        ghosts_position_max = np.empty((NUMGHOSTS, 3), dtype=int)
        for i in list(range(ghosts_position_max.shape[0])):
            ghosts_position_max[i][0] = SCREENWIDTH
            ghosts_position_max[i][1] = SCREENHEIGHT

        rewards_position_max = np.empty((SCREENHEIGHT, SCREENWIDTH), dtype=int)
        for y in list(range(rewards_position_max.shape[0])):
            for x in list(range(rewards_position_max.shape[1])):
                rewards_position_max[y][x] = FRUITREWARD

        self.observation_space = spaces.Dict(
            {
                "pacman_position": spaces.Box(
                    0, np.array([SCREENWIDTH, SCREENHEIGHT]), dtype=int
                ),
                "ghosts_position": spaces.Box(
                    0, ghosts_position_max, (NUMGHOSTS, 3), dtype=int
                ),
                "rewards_position": spaces.Box(
                    0, rewards_position_max, (SCREENHEIGHT, SCREENWIDTH), dtype=int
                ),
            }
        )
        self.action_space = spaces.Discrete(5, start=-2)

        self._pacman_position = np.array([0, 0])
        self._ghosts_position = np.zeros((NUMGHOSTS, 3), dtype=int)
        self._rewards_position = np.zeros((SCREENHEIGHT, SCREENWIDTH), dtype=int)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        if self.render_mode == "human":
            self.window = self.game.screen
            self.clock = self.game.clock

    def _getobs(self):
        self._action_to_direction = self.game.pacman.directions
        self._pacman_position = np.array(self.game.pacman.position.asInt(), int)

        for i in list(range(self._ghosts_position.shape[0])):
            self._ghosts_position[i][:2] = np.array(
                self.game.ghosts.ghosts[i].position.asInt(), int
            )
            if self.game.ghosts.ghosts[i].mode.current == FREIGHT:
                self._ghosts_position[i][2] = 1
            else:
                self._ghosts_position[i][2] = 0

        # place pellets
        for pellet in self.game.pellets.pelletList:
            self._rewards_position[pellet.position.y][pellet.position.x] = PELLETREWARD
        # place power pellets
        for pellet in self.game.pellets.powerpellets:
            self._rewards_position[pellet.position.y][pellet.position.x] = POWERPELLETREWARD
        # place fruit if exists
        if self.game.fruit != None:
            self._rewards_position[self.game.fruit.position.y][self.game.fruit.position.x] = FRUITREWARD
           

        print({
            "pacman_position": self._pacman_position,
            "ghosts_position": self._ghosts_position,
            "rewards_position": self._rewards_position,
        })

        return {
            "pacman_position": self._pacman_position,
            "ghosts_position": self._ghosts_position,
            "rewards_position": self._rewards_position,
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.game.restartGame()

        observation = self._getobs()
        info = {}
        return observation, info

    def step(self, action):
        if self.render_mode == "human":
            self.game.update(action, render=True, clocktick=self.metadata["render_fps"])
        else:
            self.game.update(
                action, render=False, clocktick=self.metadata["render_fps"]
            )

        terminated = self.game.gameOver
        reward = self.game.RLreward
        observation = self._getobs()
        info = {}

        return observation, reward, terminated, False, info

    def render(self):
        if self.render_mode == "human":
            self.game.render()

    def close(self):
        if self.window is not None:
            pygame.event.post(pygame.event.Event(QUIT))


if __name__ == "__main__":
    env = gym.make("pacman-v0", render_mode="human")
    print("Checking Environment")
    #check_env(env.unwrapped)
    #print("done checking environment")

    obs = env.reset()[0]

    for i in range(1000):
        randaction = env.action_space.sample()
        env.render()
        obs, reward, terminated, _, _ = env.step(randaction)
