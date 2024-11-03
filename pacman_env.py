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
from tqdm import tqdm
import time
import random

NOREWARD = 0
PELLETREWARD = 1
POWERPELLETREWARD = 2
FRUITREWARD = 3

GHOST_MODES = {SCATTER: 0, CHASE: 0, FREIGHT: 1, SPAWN: 2}


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
            ghosts_position_max[i][2] = GHOST_MODES[SPAWN]

        rewards_position_max = np.empty((200, 3), dtype=int)
        for y in list(range(rewards_position_max.shape[0])):
            rewards_position_max[y] = [SCREENWIDTH, SCREENHEIGHT, FRUITREWARD]

        self.observation_space = spaces.Dict(
            {
                "pacman_position": spaces.Box(
                    0, np.array([SCREENWIDTH, SCREENHEIGHT]), dtype=int
                ),
                "ghosts_position": spaces.Box(
                    0, ghosts_position_max, (NUMGHOSTS, 3), dtype=int
                ),
                "rewards_position": spaces.Box(
                    0, rewards_position_max, (200, 3), dtype=int
                ),
            }
        )
        self.action_space = spaces.Discrete(5, start=-2)

        self._pacman_position = np.array([0, 0])
        self._ghosts_position = np.zeros((NUMGHOSTS, 3), dtype=int)
        self._rewards_position = np.zeros((200, 3), dtype=int)
        self._fruit_position = None

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
            self._ghosts_position[i][2] = GHOST_MODES[
                self.game.ghosts.ghosts[i].mode.current
            ]

        # place pellets
        row = 0
        for pellet in self.game.pellets.pelletList:
            self._rewards_position[row] = np.array(
                [pellet.position.x, pellet.position.y, PELLETREWARD], int
            )
            row += 1
        # place power pellets
        for pellet in self.game.pellets.powerpellets:
            self._rewards_position[row] = np.array(
                [pellet.position.x, pellet.position.y, POWERPELLETREWARD], int
            )
            row += 1
        # place fruit if exists
        if self.game.fruit != None and self._fruit_position is None:
            self._fruit_position = self.game.fruit.position.copy()
            self._rewards_position[row] = np.array(
                [int(self._fruit_position.x), int(self._fruit_position.y), FRUITREWARD],
                int,
            )
        elif self.game.fruit == None and not (self._fruit_position is None):
            self._rewards_position[row] = np.array(
                [int(self._fruit_position.x), int(self._fruit_position.y), NOREWARD],
                int,
            )
            self._fruit_position = None

        # print(self._pacman_position.shape , self._ghosts_position.shape , self._rewards_position.shape)
        """
        for y in list(range(self._rewards_position.shape[0])):
            for x in list(range(self._rewards_position.shape[1])):
                if self._rewards_position[y][x] not in [0]:
                    print(x, y)
                    print(self._rewards_position[y][x])
        """
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

        terminated = self.game.gameOver
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


if __name__ == "__main__":
    env_not_render = gym.make("pacman-v0")
    env_render = gym.make("pacman-v0", render_mode="human")
    env = env_not_render

    model = DQN_model()
    EPISODES = 20_000

    EPSILON = 1
    EPSILON_DECAY = 0.99975
    MIN_EPSILON = 0.001

    SHOW = True
    SHOW_EVERY = 50
    RENDER = None
    EPISODES_REWARDS = []
    MIN_REWARD = (
        10 * 120 - 50 * 5
    )  # eaten half of the pellets before being eaten 5 times by ghosts

    for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit="episodes"):
        # model.tensorboard.step = episode
        if SHOW and not episode % SHOW_EVERY:
            env = env_render
        else:
            env = env_not_render

        curr_state, info = env.reset()
        curr_state = get_model_obs(curr_state)
        episode_reward = 0
        done = False

        while not done:
            # print("done: " , done)
            if np.random.random() > EPSILON:
                action = model.get_qs(curr_state)
                action = np.argmax(action)
                action -= 2
            else:
                action = random.randint(-2, 2)

            new_state, reward, terminated, truncated, info = env.step(action=action)
            new_state = get_model_obs(new_state)
            done = terminated

            episode_reward += reward

            model.update_replay_memory((curr_state, action, new_state, reward, done))
            if done:
                # print("done: " , done)
                model.train()
                # print("**********************training**********************")

            curr_state = new_state

        EPISODES_REWARDS.append(episode_reward)

        if not episode % SHOW_EVERY:
            avg_reward = sum(EPISODES_REWARDS[-SHOW_EVERY:]) / len(
                EPISODES_REWARDS[-SHOW_EVERY:]
            )
            min_reward = min(EPISODES_REWARDS[-SHOW_EVERY:])
            max_reward = max(EPISODES_REWARDS[-SHOW_EVERY:])
            # model.tensorboard.update_stats(reward_avg=avg_reward, reward_min=min_reward, reward_max=max_reward, epsilon=EPSILON)

            if min_reward >= MIN_REWARD:
                model.model.save(
                    f"models/{model.MODEL_NAME}__{max_reward:_>7.2f}max_{avg_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model"
                )

        # print("epsilon: " , EPSILON)
        if EPSILON > MIN_EPSILON:
            EPSILON *= EPSILON_DECAY
            EPSILON = max(MIN_EPSILON, EPSILON)
