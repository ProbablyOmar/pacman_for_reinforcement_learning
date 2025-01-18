import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import pygame
from pygame.locals import *
import gymnasium as gym
from gymnasium.envs.registration import register
from gymnasium.utils.env_checker import check_env
import numpy as np
from gymnasium import spaces
from run import GameController
from constants import *
from DQN_model import *
from stable_baselines3 import DQN , PPO
from modified_tensorboard import TensorboardCallback
from stable_baselines3.dqn import MultiInputPolicy
from torch.optim import RMSprop, Adam
import copy

GHOST_MODES = {SCATTER: 0, CHASE: 0, FREIGHT: 1, SPAWN: 2}


if "pacman-v0" not in gym.envs.registry:
    register(id="pacman-v0", entry_point="pacman_env:PacmanEnv", max_episode_steps=1000)


class PacmanEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self, render_mode=None , mode = SAFE_MODE , move_mode = DISCRETE_STEPS_MODE, clock_tick = 0 , pacman_lives = 1 , maze_mode = MAZE3 , pac_pos_mode = RANDOM_PAC_POS):

        self.game = GameController(rlTraining = True , mode = mode , move_mode = move_mode , clock_tick = clock_tick , pacman_lives = pacman_lives , maze_mode=maze_mode , pac_pos_mode = pac_pos_mode)
        self.num_pellets_last = 0
        self.game_score = 0
        self.useless_steps = 0
        self.episode_steps = 0

        self.num_frames_obs = 4
        reward_obs = [PELLET_MAZE , PP_MAZE , FRUIT_MAZE , GCC_MAZE , GCF_MAZE , GSC_MAZE , GSF_MAZE]

        low = np.zeros((3, GAME_ROWS, GAME_COLS), dtype=np.int_)
        high = np.zeros((3, GAME_ROWS, GAME_COLS), dtype=np.int_)

        # First matrix: 0 or 1
        low[0, :, :] = 0
        high[0, :, :] = 1

        # Second matrix: 0 or 1
        low[1, :, :] = 0
        high[1, :, :] = 1

        # Third matrix: Specific range
        low[2, :, :] = min(reward_obs)  # Minimum value in the constants
        high[2, :, :] = max(reward_obs)  # Maximum value in the constants
        self.observation_space = spaces.Box(
                    low = low , 
                    high = high , 
                    shape = (3 , GAME_ROWS , GAME_COLS) , 
                    dtype=np.int_
                )
        
        self.action_space = spaces.Discrete(4, start=0)

        self._maze_map = np.zeros(shape=(GAME_ROWS , GAME_COLS), dtype=np.int_)
        self._last_obs = np.zeros(shape=(GAME_ROWS , GAME_COLS), dtype=np.int_)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        if self.render_mode == "human":
            self.window = self.game.screen
            self.clock = self.game.clock

    def _getobs(self):
        self._maze_map = self.game.maze_map
        #self._maze_map = np.expand_dims(self._maze_map , axis=0)
        return self._maze_map

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.game.restartGame()
        self.game.done = False
        self.game_score = 0
        self.episode_steps = 0
        observation = self._getobs()
        #obs_buf = np.expand_dims(self.observation_buffer , axis=0) 
        info = {}
        return observation, info

    def step(self, action):
        if self.game.move_mode == CONT_STEPS_MODE:
            if action == 0:
                action = RIGHT
            elif action == 1:
                action = DOWN
            elif action == 2:
                action = UP
            elif action == 3:
                action = LEFT

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
                    self.episode_steps +=1
                    if terminated:
                        self.episode_steps = 0
                    return observation, step_reward, terminated, truncated, info 


        elif self.game.move_mode == DISCRETE_STEPS_MODE:
            if action == 0:
                action = RIGHT
            elif action == 1:
                action = DOWN
            elif action == 2:
                action = UP
            elif action == 3:
                action = LEFT
            #step_reward = TIME_PENALITY
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
                    self.useless_steps += 1
                    if self.useless_steps >= MAX_USELESS_STEPS:
                        self.game.done = True
                        terminated = self.game.done
                        self.useless_steps = 0
                else:
                    self.useless_steps = 0
            # if reward > 0:
            #     print(reward)

            #obs_buf = np.expand_dims(self.observation_buffer , axis=0)
            # print("***********")
            # print(reward)
            # print(terminated)
            # print("episode steps: " , self.episode_steps)
            self.episode_steps +=1
            return observation, reward, terminated, truncated, info


    def render(self):
        if self.render_mode == "human":
            self.game.render()

    def close(self):
        if self.window is not None:
            pygame.event.post(pygame.event.Event(QUIT))


if __name__ == "__main__":
    env_not_render = gym.make("pacman-v0", max_episode_steps = 10_000 ,  mode = SCARY_2_MODE , move_mode = DISCRETE_STEPS_MODE, clock_tick = 0 , pacman_lives = 7 , maze_mode = SMALL_MAZE ,  pac_pos_mode = RANDOM_PAC_POS )
    env_render = gym.make("pacman-v0", max_episode_steps = 10_000 , render_mode = "human" , mode = SCARY_1_MODE , move_mode = DISCRETE_STEPS_MODE, clock_tick = 10 , pacman_lives = 7,  maze_mode = SMALL_MAZE , pac_pos_mode = RANDOM_PAC_POS)
    
    model_path = "./models/2_ghosts_3_ch_obs"

    log_path = "./logs/fit"
   
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    if not os.path.exists(model_path):  
        os.makedirs(model_path) 
        env = env_render
        obs , _ = env.reset()

        optimizer_kwargs = dict(
            betas = (0.95, 0.999),  # Correct format: tuple for beta1 and beta2
            eps = 1e-8,  # Small epsilon to prevent division by zero
        )

        policy_kwargs = dict(
            features_extractor_class=Updated_CustomCNN_2,
            optimizer_kwargs=optimizer_kwargs,
            features_extractor_kwargs=dict(features_dim=256),
            optimizer_class=Adam,  # Using Adam optimizer here
        )

        model = DQN(
            "CnnPolicy",
            env,
            learning_rate=0.00025,
            buffer_size=10_000,
            learning_starts=10_000,
            batch_size=32,
            gamma=0.99,
            train_freq=(4, "step"),
            gradient_steps=2,
            target_update_interval=500,
            exploration_fraction=0.1,
            exploration_initial_eps=1,
            exploration_final_eps=0.5,
            policy_kwargs=policy_kwargs,
            verbose=1,
            tensorboard_log=log_path,
            device='cuda',
        )

        #print("here ***********: " , model.exploration_fraction , model.exploration_initial_eps , model.exploration_final_eps , model.policy)
        time_steps = 1000000
        for i in range (50):
            model.learn(total_timesteps = time_steps , progress_bar=True , reset_num_timesteps = False , tb_log_name = "./cnn/2_ghosts_3_ch_obs")
            model.save(f"{model_path}/{(i+1)*time_steps}") 

    elif os.path.exists(model_path):
        env = env_render
        obs , _ = env.reset()
        model_final_path = f"./{model_path}/3000000.zip"
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
#     env = gym.make("pacman-v0", max_episode_steps = 10_000 , render_mode = "human" , mode = SCARY_2_MODE , move_mode = DISCRETE_STEPS_MODE, clock_tick = 10 , pacman_lives = 3,  maze_mode = RAND_MAZE ,  pac_pos_mode = RANDOM_PAC_POS )
#     # print("Checking Environment")
#     # check_env(env.unwrapped)
#     # print("done checking environment")

#     obs = env.reset()[0]
#     done = False
#     action = 4
#     num_steps = 1
#     while not done:
#         # if num_steps == 10:
#         #     break
#         randaction = env.action_space.sample()
#         env.render()
#         obs, reward, terminated, _, _ = env.step(action)
#         done = terminated 
        
#         # print("***************************************")
#         # print(obs.shape)
#         # print(obs[0][0])
#         # print(obs[0][1])
#         # print(obs[0][2])
#         # print(obs[0][3])
#         print(reward)
#         # if action == 4:
#         #     action = 0
#         # elif action == 0:
#         #     action = 4

#         # num_steps +=1
#         # if num_steps > 10:
#         #     break
#         # #print(env.game_score)
#         # if action == 1 and reward == HIT_WALL_PENALITY:
#         #     #print("*****************here")
#         #     action = 2
#         # elif reward == HIT_WALL_PENALITY:
#         #     action = 1



















   







