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
from multi_ddpg import MADDPG
from buffer import MultiAgentReplayBuffer
import numpy as np



GHOST_MODES = {SCATTER: 0, CHASE: 0, FREIGHT: 1, SPAWN: 2}


if "pacman-v0" not in gym.envs.registry:
    register(id="pacman-v0", entry_point="pacman_env:PacmanEnv", max_episode_steps=1000)


class PacmanEnv(ParallelEnv):
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self, render_mode=None, mode = SCARY_1_MODE , move_mode = DISCRETE_STEPS_MODE, clock_tick = 10 , pacman_lives = 3 , maze_mode = MAZE1 , pac_pos_mode = NORMAL_PAC_POS):
        self.game = GameController(rlTraining=True, mode = mode , move_mode = move_mode , clock_tick = clock_tick , pacman_lives = pacman_lives , maze_mode=maze_mode , pac_pos_mode = pac_pos_mode)
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
        if agent == "pacman" or "ghost":
            return spaces.Box(
                low=0, high=13, shape=(1, GAME_ROWS, GAME_COLS), dtype=np.int_
            )
        # elif agent == "ghost":
        #     return spaces.Box(0, np.array([SCREENWIDTH, SCREENHEIGHT]), dtype=int)

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return spaces.Discrete(5)

    def _getobs(self):
        # print("Pacman observation:" , self.game.maze_map)  
        # print("Pacman position:", self.game.pacman.position)  
        # self._maze_map = self.game.observation
        self._maze_map = self.game.maze_map
        self._maze_map = np.expand_dims(self._maze_map, axis=0)
        
        
        # ghost_positions = [np.array([ghost.position.x, ghost.position.y]) for ghost in self.game.ghosts]
        
        observations = {
            "pacman": self._maze_map,              
            "ghost": self._maze_map
            }
    
        return observations


    def reset(self, seed=None, options=None):
        self.agents = copy.copy(self.possible_agents)
        # print("Possible Agents",self.possible_agents)
        self.game.restartGame()

        observation = self._getobs()
        info = {"pacman": {}, "ghost": {}}
        
        return observation, info

    def step(self, agents_directions):
        
        # agents_directions = {"pacman" : RIGHT , "ghosts" : [None , None , None , None]}
        # pacman_action = agents_directions["pacman"]
        # ghost_action = agents_directions["ghosts"]
        
        # print("agents_directions:", agents_directions)
        # print("agents_directions: pacman",agents_directions["pacman"].shape)
        agents_directions["pacman"] = agents_directions["pacman"].squeeze(0).argmax(dim=-1).item()
        pacman_action = None 
        if self.game.move_mode == DISCRETE_STEPS_MODE:
            if agents_directions["pacman"] == 0:  # Right
                pacman_action = RIGHT
            elif agents_directions["pacman"] == 1:  # Down
                pacman_action = DOWN
            elif agents_directions["pacman"] == 2:  # Up
                pacman_action = UP
            elif agents_directions["pacman"] == 3:  # Left
                pacman_action = LEFT
            # else:
            #     print("Pacman actions provided" , agents_directions["pacman"])
            
        
            # print("Observation space for pacman:", env.observation_space("pacman"))
            # print("Action space for pacman:", env.action_space("pacman"))
            # print("Observation space for ghosts:", env.observation_space("ghosts"))
            # print("Action space for ghosts:", env.action_space("ghosts"))
            
            # Process actions for ghost
            ghost_action = None
            if "ghosts" in agents_directions:
                if agents_directions["ghosts"][0] == 0:  # Right
                    ghost_action = RIGHT
                elif agents_directions["ghosts"][0] == 1:  # Down
                    ghost_action = DOWN
                elif agents_directions["ghosts"][0] == 2:  # Up
                    ghost_action = UP
                elif agents_directions["ghosts"][0] == 3:  # Left
                    ghost_action = LEFT
            # else:
            #     # Handle the case where "ghosts" is not in agents_directions
            #     print("ghost actions recieved.", agents_directions["ghost"])
 
                
            agents_directions = {
                    "pacman": pacman_action,  
                    "ghosts": [ghost_action , None , None , None] 
                }
                
            # Execute the game update
            if self.render_mode == "human":
                self.game.update(
                    agents_directions=agents_directions,
                    render=True
                )
            else:
                self.game.update(
                    agents_directions=agents_directions,
                    render=False
                )

            # Calculate rewards based on Pacman and Ghosts' positions
            pacman_position = self.game.pacman.position
            ghost_positions = [ghost.position for ghost in self.game.ghosts]
            
            # Example reward logic:
            pacman_reward = self.game.RLreward  # Reward from Pacman's actions
            ghost_reward = 0
            for ghost_position in ghost_positions:
                distance = math.hypot(pacman_position.x - ghost_position.x, pacman_position.y - ghost_position.y)
                if distance == 0:
                    distance = 1e-6 #avoid div by 0
                    ghost_reward += (1 / distance) * 50  # Simple reward based on distance from Pacman

            # Update terminated and truncated flags
            terminated = {a: self.game.done for a in self.agents}
            truncated = {a: False for a in self.agents}

            # Return observations and rewards
            observations = self._getobs()
            reward = {"pacman": pacman_reward, "ghost": ghost_reward}
            info = {a: {} for a in self.agents}
            
            if reward != TIME_PENALITY:
                step_reward = reward

            if not np.array_equal(observations["pacman"], self._last_obs):
                np.copyto(self._last_obs, observations["pacman"])
                self.game_score += step_reward["pacman"]
                
            # print("****************************")
            # print("Observations:", observations)
            # print("Rewards:", reward)
            # print("Terminated:", terminated)
            # print("Truncated:", truncated)

            return observations, step_reward, terminated, truncated, info 


    def render(self):
        if self.render_mode == "human":
            self.game.render()

    def close(self):
        if self.window is not None:
            pygame.event.post(pygame.event.Event(QUIT))

#################################################################
#train maddpg 

def obs_list_to_state_vector(obs_list):
    state = np.array([])
    for obs in obs_list:
        # Flatten obs to 1D if it has more than 1 dimension
        obs = np.ravel(obs) if isinstance(obs, np.ndarray) else np.array([obs])
        state = np.concatenate([state, obs])
    return state



if __name__ == "__main__":
    

    env_not_render = PacmanEnv(render_mode=None,  mode = SCARY_1_MODE , move_mode = DISCRETE_STEPS_MODE, clock_tick = 0 , pacman_lives = 3 , maze_mode = MAZE1 ,  pac_pos_mode = NORMAL_PAC_POS )
    env_render = PacmanEnv(render_mode="human" ,  mode = SCARY_1_MODE , move_mode = DISCRETE_STEPS_MODE, clock_tick = 10 , pacman_lives = 3 , maze_mode = MAZE1 ,  pac_pos_mode = NORMAL_PAC_POS )


    model_path = "./models/maddpg"
    log_path = "./logs/maddpg"
    chkpt_dir = "./tmp/maddpg/"

    os.makedirs(model_path, exist_ok=True)
    os.makedirs(log_path, exist_ok=True)
    os.makedirs(chkpt_dir, exist_ok=True)

    
    # env = env_not_render
    env = env_render
    #n_agents = 2
    possible_agents = env.possible_agents 
    
    
    actor_dims = []
    for agent in env.possible_agents:
        if agent == "pacman":
            actor_dims.append(np.prod(env.observation_space(agent).shape))  # Flattening the observation space
        elif agent == "ghost":
            actor_dims.append(np.prod(env.observation_space(agent).shape))  # Flattening the observation space
            
    joint_state_dim = sum(actor_dims)  # Sum of state dimensions for all agents
    joint_action_dim = sum([env.action_space(agent).n for agent in env.possible_agents])  
    
    critic_dims = [joint_state_dim, joint_action_dim]
    
    
    
    n_actions = 5
    # Debugging output

    # print("Possible agents:", possible_agents)
    # print("Actor dimensions:", actor_dims)
    # print("Critic dimensions:", critic_dims)
    
    

    model_final_path = os.path.join(model_path, "MADDPG_model.pth")
    if not os.path.exists(model_final_path):
        print("Training new MADDPG model...")
        
        maddpg_agents = MADDPG(
            actor_dims,  # Actor; Takes individual states
            critic_dims,  # Critic; Takes joint states and joint actions 
            n_agents=len(env.possible_agents),
            n_actions=n_actions, 
            possible_agents=env.possible_agents,
            env=env,
            alpha=0.01,
            beta=0.01,
            gamma=0.99,
            tau=0.01,
            chkpt_dir=chkpt_dir,
            tensorboard_log=log_path,
            device='cuda',
            )

            
        memory = MultiAgentReplayBuffer(
                max_size=100000, 
                critic_dims=critic_dims,  
                actor_dims=actor_dims,  
                possible_agents=possible_agents,
                n_agents=len(env.possible_agents), 
                n_actions=n_actions, 
                batch_size=1024,
                env=env
            )


        total_episodes = 5000
        MAX_STEPS = 100
        n_agents = len(possible_agents) 
        PRINT_INTERVAL = 100
        total_steps = 0
        best_score = -np.inf
        score_history = []
        evaluate = False
        
        
        if evaluate:
            maddpg_agents.load_checkpoint()
            
        
        for episode in range(total_episodes):
            obs, _ = env.reset()  
            score = 0
            done = [False] * n_agents
            episode_step = 0

            while not any(done):
                if evaluate:
                    env.render()
                # Chooses actions for each agent based on their individual states
                # ##### Actor
                actions = {}
                for agent in env.possible_agents:
                    raw_obs = [obs[agent] for agent in env.possible_agents]  # Collect all agent observations
                    # print("Raw observations:", raw_obs)
                    actions = maddpg_agents.choose_action(raw_obs)
                    obs_, rewards, terminated, truncated, info = env.step(actions)
                
                

                # Prepare joint state and next joint state for the critic
                state = obs_list_to_state_vector([obs[agent] for agent in env.possible_agents])
                state_ = obs_list_to_state_vector([obs_[agent] for agent in env.possible_agents])
                
                if episode_step >= MAX_STEPS:
                    done = [True]*n_agents
                
                ###store in replay buffer
                memory.store_transition(obs, state, actions, rewards, obs_, state_, done)  
                
                # Update agent after each step
                maddpg_agents.learn(memory)
                
                if total_steps % 100 == 0 and not evaluate:
                    maddpg_agents.learn(memory)
                
                obs = obs_
                
                # Extract values from the rewards dictionary and sum them
                score += sum(rewards.values())
                print(score)

                total_steps += 1
                episode_step += 1
                
            score_history.append(score)
            avg_score = np.mean(score_history[-100:])
            # Print episode summary
            print(f"Episode {episode + 1}/{total_episodes}:")
            
                
            if not evaluate:
                if avg_score > best_score:
                    maddpg_agents.save_checkpoint()  # Save the model if it performs better than before
                    best_score = avg_score
                    
                    
            if episode % PRINT_INTERVAL == 0 and episode > 0:
                print('episode', episode, 'average score {:.1f}'.format(avg_score))
                
        env.close()