import pygame
from pygame.locals import *
import gymnasium as gym
from gymnasium.envs.registration import register
from gymnasium.utils.env_checker import check_env
import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv
from run import GameController
from constants import *
from DQN_model import CustomCNN
from stable_baselines3 import DQN
from stable_baselines3.dqn import MultiInputPolicy
from pettingzoo.test import parallel_api_test
import os
import copy
import functools

GHOST_MODES = {SCATTER: 0, CHASE: 0, FREIGHT: 1, SPAWN: 2}


if "pacman-v0" not in gym.envs.registry:
    register(id="pacman-v0", entry_point="pacman_env:PacmanEnv", max_episode_steps=1000)


class PacmanEnv(ParallelEnv):
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self, render_mode=None):
        self.game = GameController(rlTraining=True, mode=SAFE_MODE)
        self.game_score = 0
        self.useless_steps = 0
        self.possible_agents = ["pacman", "ghost"]

        self._maze_map = np.zeros(shape=(GAME_ROWS, GAME_COLS), dtype=np.int_)
        self._last_obs = np.zeros(shape=(GAME_ROWS, GAME_COLS), dtype=np.int_)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        if self.render_mode == "human":
            self.window = self.game.screen
            self.clock = self.game.clock

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        if agent == "pacman":
            return spaces.Box(
                low=0, high=13, shape=(1, GAME_ROWS, GAME_COLS), dtype=np.int_
            )
        elif agent == "ghost":
            return spaces.Box(0, np.array([SCREENWIDTH, SCREENHEIGHT]), dtype=int)

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return spaces.Discrete(5, start=0)

    def _getobs(self):
        self._maze_map = self.game.observation
        self._maze_map = np.expand_dims(self._maze_map, axis=0)
        observations = {"pacman": self._maze_map, "ghost": self.game.pacman.position}

        return observations

    def reset(self, seed=None, options=None):
        self.agents = copy.copy(self.possible_agents)
        self.game.restartGame()

        observation = self._getobs()
        info = {"pacman": "", "ghost": ""}
        return observation, info

    def step(self, actions):
        pacman_action = actions["pacman"]
        ghost_action = actions["ghost"]
        pacman_action -= 2
        step_reward = TIME_PENALITY
        while True:
            if self.render_mode == "human":
                self.game.update(
                    agent_direction=pacman_action,
                    render=True,
                    clocktick=self.metadata["render_fps"],
                )
            else:
                self.game.update(
                    agent_direction=pacman_action,
                    render=False,
                    clocktick=self.metadata["render_fps"],
                )

            terminated = {a: self.game.done for a in self.agents}
            truncated = {a: False for a in self.agents}
            reward = {"pacman": self.game.RLreward, "ghost": 0}
            observations = self._getobs()
            info = {a: {} for a in self.agents}

            if reward != TIME_PENALITY:
                step_reward = reward

            if not np.array_equal(observations["pacman"], self._last_obs):
                np.copyto(self._last_obs, observations["pacman"])
                self.game_score += step_reward["pacman"]

                if self.game.mode == SAFE_MODE:
                    if reward == TIME_PENALITY or reward == HIT_WALL_PENALITY:
                        self.useless_steps += 1
                        if self.useless_steps >= MAX_USELESS_STEPS:
                            self.game.done = True
                            terminated = {a: self.game.done for a in self.agents}
                            self.agents = []
                            self.useless_steps = 0
                    # else:
                    #     self.useless_steps = 0
                return observations, step_reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "human":
            self.game.render()

    def close(self):
        if self.window is not None:
            pygame.event.post(pygame.event.Event(QUIT))


"""
if __name__ == "__main__":
    env = PacmanEnv()
    parallel_api_test(env, num_cycles=1_000)
"""


if __name__ == "__main__":
    env_not_render = gym.make("pacman-v0", max_episode_steps=10_000)
    env_render = gym.make("pacman-v0", max_episode_steps=10_000, render_mode="human")

    model_path = "./models/dqn_baseline_cnn3"  # with max_useless_steps = 1000
    log_path = "./logs/fit"

    if not os.path.exists(log_path):
        os.makedirs(log_path)

    if not os.path.exists(model_path):
        print("can't find path")
        os.makedirs(model_path)
        env = env_not_render
        obs, _ = env.reset()

        policy_kwargs = dict(
            features_extractor_class=CustomCNN,
            features_extractor_kwargs=dict(features_dim=256),
        )

        model = DQN(
            "CnnPolicy",
            env,
            learning_rate=0.0001,
            learning_starts=2000,
            batch_size=32,  # 32
            gamma=0.97,
            # train_freq = (1, "episode"),
            gradient_steps=4,
            target_update_interval=100,
            exploration_fraction=0.99,
            exploration_initial_eps=1,
            exploration_final_eps=0.1,
            policy_kwargs=policy_kwargs,
            verbose=1,
            tensorboard_log=log_path,
        )

        time_steps = 400_000
        for i in range(100):
            model.learn(
                total_timesteps=time_steps,
                progress_bar=True,
                reset_num_timesteps=False,
                tb_log_name="./cnn/dqn_baseline_cnn_alex_net",
            )
            model.save(f"{model_path}/{(i+1)*time_steps}")

    elif os.path.exists(model_path):
        env = env_render
        obs, _ = env.reset()
        model_final_path = f"{model_path}/1200000.zip"
        model = DQN.load(model_final_path, env=env)
        print(model.policy)

        episodes = 10
        for ep in range(episodes):
            done = False
            while not done:
                action, next_state = model.predict(obs)
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
