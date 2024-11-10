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


if __name__ == "__main__":
    env_not_render = gym.make("pacman-v0", max_episode_steps = 500)
    env_render = gym.make("pacman-v0", render_mode = "human")
    env = env_not_render

    model = DQN_model()
    EPISODES = 20_000
    #EPISODES = 2

    EPSILON = 1
    EPSILON_DECAY = 0.997
    MIN_EPSILON = 0.001

    SHOW = True
    SHOW_EVERY = 2
    #SHOW_EVERY = 1

    RENDER = None
    EPISODES_REWARDS = []
    min_avg_reward = -1000

    min_rews = []
    max_rews = []
    avg_rews = []
    num_steps = []
    episode_num = 0

    save_dir = "/graphs"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit="episodes"):
        episode_num +=1
        model.tensorboard.step = episode
        if SHOW and not episode % SHOW_EVERY:
            env = env_render
        else:
            env = env_not_render

        curr_state, info = env.reset()
        curr_state = get_model_obs(curr_state)
        episode_reward = 0
        done = False
        steps = 0

        while not done:
            steps += 1
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

        num_steps.append(steps)
        EPISODES_REWARDS.append(episode_reward)

        if not episode % SHOW_EVERY:
            avg_reward = sum(EPISODES_REWARDS[-SHOW_EVERY:]) / len(
                EPISODES_REWARDS[-SHOW_EVERY:]
            )
            min_reward = min(EPISODES_REWARDS[-SHOW_EVERY:])
            max_reward = max(EPISODES_REWARDS[-SHOW_EVERY:])
            model.tensorboard.update_stats(reward_avg=avg_reward, reward_min=min_reward, reward_max=max_reward, epsilon=EPSILON)

            min_rews.append(min_reward)
            max_rews.append(max_reward)
            avg_rews.append(avg_reward)

            print("Episode: ** " , episode_num)
            print("Avg reward: " , avg_reward)
            print("min reward: " , min_reward)
            print("max reward: " , max_reward)


            if avg_reward >= min_avg_reward:
                model.model.save(
                    f"models/{model.MODEL_NAME}__{max_reward:_>7.2f}max_{avg_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.keras"
                )
                min_avg_reward = avg_reward

            #########################plotting#############################
            episodes_divs = np.arange(SHOW_EVERY, episode_num + 1, SHOW_EVERY)
            episodes = np.arange(1, episode_num + 1)

            plt.figure(figsize=(12, 8))

            plt.plot(episodes_divs, avg_rews, label="Average Reward", color="blue")
            plt.plot(episodes_divs, min_rews, label="Minimum Reward", color="red")
            plt.plot(episodes_divs, max_rews, label="Maximum Reward", color="green")
            plt.plot(episodes, num_steps, label="Number Of Steps", color="black")

            plt.xlabel("Episode")
            plt.title("Reward Metrics Every 50 Episodes")
            plt.legend(loc="best")
            plot_filename = os.path.join(save_dir, "reward_metrics.png")
            plt.savefig(plot_filename)
            plt.show()
            #################################################################

        # print("epsilon: " , EPSILON)
        if EPSILON > MIN_EPSILON:
            EPSILON *= EPSILON_DECAY
            EPSILON = max(MIN_EPSILON, EPSILON)







