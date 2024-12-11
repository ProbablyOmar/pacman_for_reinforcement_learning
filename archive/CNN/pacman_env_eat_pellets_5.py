import pygame
from pygame.locals import *
import gymnasium as gym
from gymnasium.envs.registration import register
from gymnasium.utils.env_checker import check_env
import numpy as np
from gymnasium import spaces
from run import GameController
from constants import *
from DQN_model import CustomCNN , CustomCNN_eat_pellets2 , CustomCNN_eat_pellets3 , CustomCNN_eat_pellets5
from stable_baselines3 import DQN
from modified_tensorboard import TensorboardCallback
from stable_baselines3.dqn import MultiInputPolicy
import os
import copy

GHOST_MODES = {SCATTER: 0, CHASE: 0, FREIGHT: 1, SPAWN: 2}


if "pacman-v0" not in gym.envs.registry:
    register(id="pacman-v0", entry_point="pacman_env:PacmanEnv", max_episode_steps=1000)


class PacmanEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self, render_mode=None , mode = SAFE_MODE , move_mode = DISCRETE_STEPS_MODE, clock_tick = 0 , pacman_lives = 1):

        self.game = GameController(rlTraining = True , mode = mode , move_mode = move_mode , clock_tick = clock_tick , pacman_lives = pacman_lives)
        self.num_pellets_last = 0
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
        self.game_score = 0

        observation = self._getobs()
        info = {}
        return observation, info

    def step(self, action):
        if self.game.move_mode == CONT_STEPS_MODE:
            action -= 2
            step_reward = TIME_PENALITY
            while True:
                if self.render_mode == "human":
                    self.game.update(
                        agent_direction=action,
                        render=True
                        #clocktick=self.metadata["render_fps"],
                    )
                else:
                    self.game.update(
                        agent_direction=action,
                        render=False
                        #clocktick=self.metadata["render_fps"],
                    )
                
                terminated = self.game.done
                truncated = False
                reward = self.game.RLreward
                observation = self._getobs()
                info = {}

                if reward != TIME_PENALITY:
                    step_reward = reward

                if not np.array_equal(observation , self._last_obs): 
                    self.num_pellets_last = len(self.game.pellets.pelletList)
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


        elif self.game.move_mode == DISCRETE_STEPS_MODE:
            action -= 2
            step_reward = TIME_PENALITY
            #while True:
            if self.render_mode == "human":
                self.game.update(
                    agent_direction=action,
                    render=True
                    #clocktick=self.metadata["render_fps"],
                )
            else:
                self.game.update(
                    agent_direction=action,
                    render=False
                    #clocktick=self.metadata["render_fps"],
                )
            self.num_pellets_last = len(self.game.pellets.pelletList)
            terminated = self.game.done
            truncated = False
            reward = self.game.RLreward
            observation = self._getobs()
            info = {}

            #if not np.array_equal(observation , self._last_obs): 
            #np.copyto(self._last_obs , observation)
            self.game_score += reward

            if self.game.mode == SAFE_MODE:
                if reward == TIME_PENALITY or reward == HIT_WALL_PENALITY:
                    self.useless_steps +=1
                    if self.useless_steps >= MAX_USELESS_STEPS:
                        self.game.done = True
                        terminated = self.game.done
                        self.useless_steps = 0
                # else:
                #     self.useless_steps = 0
            return observation, reward, terminated, truncated, info


    def render(self):
        if self.render_mode == "human":
            self.game.render()

    def close(self):
        if self.window is not None:
            pygame.event.post(pygame.event.Event(QUIT))


if __name__ == "__main__":
    env_not_render = gym.make("pacman-v0", max_episode_steps = 10_000 ,  mode = SAFE_MODE , move_mode = DISCRETE_STEPS_MODE, clock_tick = 0 , pacman_lives = 1)
    env_render = gym.make("pacman-v0", max_episode_steps = 10_000 , render_mode = "human" , mode = SAFE_MODE , move_mode = DISCRETE_STEPS_MODE, clock_tick = 0 , pacman_lives = 1)
    
    model_load_path = "./models/dqn_baseline_cnn3_eat_pellets_3"   #with max_useless_steps = 1000
    model_path = "./models/dqn_baseline_cnn3_eat_pellets_5"

    log_path = "./logs/fit"
   
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    if not os.path.exists(model_path):  
        os.makedirs(model_path) 
        env = env_not_render
        obs , _ = env.reset()

        model_final_path = f"{model_load_path}/400000"

        model = DQN.load(model_final_path , env = env) 
        model.features_extractor_class = CustomCNN_eat_pellets5
        model.features_extractor_kwargs = dict(features_dim=1024),
        model.learning_rate =  0.00005
        # model.exploration_fraction = 1,
        # model.exploration_initial_eps=1,
        # model.exploration_final_eps=0.5,  ## changed in pellets_4

        time_steps = 400_000
        for i in range (100):
            model.learn(total_timesteps = time_steps , progress_bar=True , reset_num_timesteps = False , tb_log_name = "./cnn/dqn_baseline_cnn3_eat_pellets_5" , callback=TensorboardCallback(env))
            model.save(f"{model_path}/{(i+2)*time_steps}") 

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



TIME_PENALITY = -0.5  ### -.01
RAND_PENALITY = -0.08    ## this will only be in the safe mode training once ghosts are in disable it
PELLET_LOST_PENALITY = -0.08
HIT_WALL_PENALITY = -3.5 

PELLET_REWARD_UPDATE = 0.02
PELLET_REWARD = 10
POWERPELLET_REWARD = 50
FRUIT_REWARD = 200
FINISH_LEVEL_REWARD = 50


GHOST_PENALITY = -50
GHOST_REWARD = 7.5
GHOST_UPDATE_REWARD = 0.5
GHOST_UPDATE_PENALITY = 2  