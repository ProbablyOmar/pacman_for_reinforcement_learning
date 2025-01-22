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
import math
from torch.optim import Adam
# from modified_tensorboard import TensorboardCallback
from multi_ddpg import Agent
from buffer import MultiAgentReplayBuffer



GHOST_MODES = {SCATTER: 0, CHASE: 0, FREIGHT: 1, SPAWN: 2}


if "pacman-v0" not in gym.envs.registry:
    register(id="pacman-v0", entry_point="pacman_env:PacmanEnv", max_episode_steps=1000)


class PacmanEnv(ParallelEnv):
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self, render_mode=None):
        self.game = GameController(rlTraining=True)
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
        
        observations = {
            "pacman": self._maze_map,              
            "ghost": self.game.pacman.position     
            }

        #global state for the critic network
        combined_state = {
        "pacman": self._maze_map,
        "ghost": self.game.pacman.position,
        "ghost_position": [ghost.position for ghost in self.game.ghosts]
    }
        return observations, combined_state

    def reset(self, seed=None, options=None):
        self.agents = copy.copy(self.possible_agents)
        self.game.restartGame()

        observation, _ = self._getobs()
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
                    agent_directions={"pacman": pacman_action, "ghost": ghost_action},
                    render=True,
                    # clocktick=self.metadata["render_fps"],
                )
            else:
                self.game.update(
                    agent_direction={"pacman": pacman_action, "ghost": ghost_action},
                    render=False,
                    # clocktick=self.metadata["render_fps"],
                )
                
            pacman_position = self.game.pacman.position
            ghost_position = self.game.ghosts[0].position
            distance = math.hypot(pacman_position.x - ghost_position.x, pacman_position.y - ghost_position.y)
            
            ghostReward = (
                1 / distance
            ) * 50 + self.game.pacmanEaten * 500
            
            
            # ghostReward = (
            #     1 / (self.game.ghosts[0].position - self.game.pacman.position)
            # ) * 50 + self.game.pacmanEaten * 500
            
            terminated = {a: self.game.done for a in self.agents}
            truncated = {a: False for a in self.agents}
            reward = {"pacman": self.game.RLreward, "ghost": ghostReward}
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

#############################
#train maddpg 

if __name__ == "__main__":
    
    env_not_render = PacmanEnv(render_mode=None)
    env_render = PacmanEnv(render_mode="human")


    model_path = "./models/maddpg"
    log_path = "./logs/maddpg"
    chkpt_dir = "./tmp/maddpg/"

    os.makedirs(model_path, exist_ok=True)
    os.makedirs(log_path, exist_ok=True)
    os.makedirs(chkpt_dir, exist_ok=True)

    
    env = env_render
    n_agents = 2  
    actor_dims = [env.observation_space(agent).shape[0] for agent in env.possible_agents]  # Actor input dimensions
    critic_dims = sum(actor_dims)  # Critic input dimensions "joint state"
    n_actions = 5  

    
    model_final_path = os.path.join(model_path, "MADDPG_model.pth")
    if not os.path.exists(model_final_path):
        print("Training new MADDPG model...")

        
        model = Agent(
            actor_dims,  # Actor; Takes individual states
            critic_dims,  # Critic; Takes joint states and joint actions 
            n_agents,
            n_actions,
            alpha=0.01,
            beta=0.01,
            chkpt_dir=chkpt_dir,
            tensorboard_log=log_path,
            device='cuda'
        )

        
        memory = MultiAgentReplayBuffer(1000000, critic_dims, actor_dims, n_agents, n_actions, batch_size=1024)

        total_episodes = 5000
        PRINT_INTERVAL = 100
        total_steps = 0
        best_score = -np.inf
        score_history = []

        for episode in range(total_episodes):
            obs, _ = env.reset()  
            done = [False] * n_agents  
            episode_score = 0

            while not any(done):
                # Chooses actions for each agent based on their individual states
                # ##### Actor
                actions = model.choose_action(obs)  
                actions = {agent: actions[i] for i, agent in enumerate(env.possible_agents)}  
                print(f"Actions: {actions}")
                for agent in env.possible_agents:
                    print(f"{agent} action space: {env.action_space(agent)}")
                obs_, rewards, done, _ = env.step(actions)

                # Prepare joint state and next joint state for the critic
                state = np.concatenate([observations[agent] for agent in env.possible_agents])  # Joint state
                state_ = np.concatenate([obs_[agent] for agent in env.possible_agents])  # Next joint state

                ###store in replay buffer
                memory.store_transition(observations, state, actions, rewards, obs_, state_, done)  

                if total_steps > memory.batch_size:
                    #learning step; Update actor and critic networks
                    actor_loss, critic_loss = model.learn(memory)
                    print(f"Actor Loss: {actor_loss}, Critic Loss: {critic_loss}")

                
                observations = obs_
                episode_score += sum(rewards.values())
                total_steps += 1

            
            score_history.append(episode_score)
            avg_score = np.mean(score_history[-100:])
            
            if avg_score > best_score:
                best_score = avg_score
                model.save_checkpoint(path=model_final_path)

            if episode % PRINT_INTERVAL == 0:
                print(f"Episode {episode}, Average Score: {avg_score:.2f}")

    else:
        print("Loading pre-trained MADDPG model...")
        #load the pre-trained model
        model = MADDPGAgent.load_checkpoint(model_final_path)

        #run for evaluation
        episodes = 10
        for ep in range(episodes):
            observations, _ = env.reset()
            done = [False] * n_agents
            episode_score = 0

            while not any(done):
                
                actions = model.choose_action(observation=observations, explore=False)  
                obs_, rewards, done, _ = env.step(actions)
                episode_score += sum(rewards.values())
                observations = obs_

            print(f"Episode {ep + 1} - Total Score: {episode_score}")

    env.close()




# if __name__ == "__main__":
#     os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
#     env = gym.make("pacman-v0", max_episode_steps = 10_000 , render_mode = "human" , mode = SCARY_2_MODE , move_mode = DISCRETE_STEPS_MODE, clock_tick = 10 , pacman_lives = 1,  maze_mode = RAND_MAZE ,  pac_pos_mode = RANDOM_PAC_POS )
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



















   







