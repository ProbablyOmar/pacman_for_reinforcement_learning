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
        self.game = GameController(rlTraining=True)

        self.observation_space = spaces.Dict(
            {
                "maze_map": spaces.Box(
                    low = 0, high = 4 , shape = (GAME_ROWS , GAME_COLS) , dtype=np.int_
                ),
                "ghosts_position": spaces.Box(
                    low = np.array([[0,0,-2,1],
                                    [0,0,-2,1],
                                    [0,0,-2,1],
                                    [0,0,-2,1]]
                    ), 
                    high = np.array([[GAME_ROWS,GAME_COLS,2,5],
                                    [GAME_ROWS,GAME_COLS,2,5],
                                    [GAME_ROWS,GAME_COLS,2,5],
                                    [GAME_ROWS,GAME_COLS,2,5]]
                    ), 
                    shape = (4,4) , 
                    dtype=np.int_
                ),
                "pacman_position": spaces.Box(
                    low = np.array([0,0]), high = np.array([GAME_ROWS , GAME_COLS]) , dtype=np.int_
                ),
            }
        )
        self.action_space = spaces.Discrete(5, start=0)

        self._maze_map = np.zeros(shape=(GAME_ROWS , GAME_COLS), dtype=np.int_)
        self._ghosts_position = np.zeros(shape = (4,4) , dtype=np.int_)
        self._pacman_position = np.array([0, 0], dtype=np.int_)
        self._last_obs = None

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        if self.render_mode == "human":
            self.window = self.game.screen
            self.clock = self.game.clock

    def _getobs(self):
        self._pacman_position = np.array(self.game.pacman.tile)
        self._maze_map = self.game.maze_map

        for num ,  ghost in enumerate(self.game.ghosts.ghosts):
            x , y = ghost.tile
            direction = ghost.direction
            reward = GHOST_REWARDS[ghost.points]
            self._ghosts_position[num] = np.array([x , y , direction , reward])


        # print( {
        #     "maze_map": self._maze_map,
        #     "ghosts_position": self._ghosts_position,
        #     "pacman_position" :  self._pacman_position
        # })

        return {
            "maze_map": self._maze_map,
            "ghosts_position": self._ghosts_position,
            "pacman_position" :  self._pacman_position
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.game.restartGame()

        observation = self._getobs()
        info = {}
        return observation, info

    def step(self, action):
        action -= 2
        step_reward = -1
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

            if reward != -1:
                step_reward = reward
            if self._last_obs is None or not all(np.array_equal(observation[key], self._last_obs[key]) for key in observation): 
                self._last_obs = copy.deepcopy(observation)
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
    
    model_path = "./models/simple_DQN_full_maze_map_baseline_obs_updated_norm_rew3"
    log_path = "./logs/fit"
   
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    if not os.path.exists(model_path):  
        os.makedirs(model_path)
        env = env_not_render
        obs , _ = env.reset()
        
        model = DQN("MultiInputPolicy" , env ,  verbose = 1 , tensorboard_log = log_path)

        time_steps = 400_000
        for i in range (100):
            model.learn(total_timesteps = time_steps , progress_bar=True , reset_num_timesteps = False , tb_log_name = "simple_DQN_full_maze_map_baseline_obs_updated_norm_rew3")
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
                print(reward)
                done = terminated
        env.close()


# if __name__ == "__main__":
#     env = gym.make("pacman-v0", render_mode="human")
#     # print("Checking Environment")
#     # check_env(env.unwrapped)
#     # print("done checking environment")

#     obs = env.reset()[0]

#     while True:
#         randaction = env.action_space.sample()
#         env.render()
#         obs, reward, terminated, _, _ = env.step(4)
#         # print(obs)
#         # print("***************" , reward)
