import pygame
from pygame.locals import *
import gymnasium as gym
from gymnasium.envs.registration import register
from gymnasium.utils.env_checker import check_env
from gymnasium import spaces
from pettingzoo import ParallelEnv
from run import GameController
from constants import *
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
import matplotlib.pyplot as plt



GHOST_MODES = {SCATTER: 0, CHASE: 0, FREIGHT: 1, SPAWN: 2}


if "pacman-v0" not in gym.envs.registry:
    register(id="pacman-v0", entry_point="pacman_env:PacmanEnv", max_episode_steps=1000)


class PacmanEnv(ParallelEnv):
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self, render_mode=None, mode = SCARY_1_MODE , move_mode = DISCRETE_STEPS_MODE, clock_tick = 10 , pacman_lives = 3 , maze_mode = MAZE1 , pac_pos_mode = RANDOM_PAC_POS):
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
        if agent in ["pacman", "ghost"]:
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
        # agents_directions["pacman"] = agents_directions["pacman"].squeeze(0).argmax(dim=-1).item()
        pacman_action_index = agents_directions["pacman"].argmax(dim=-1).item()
        pacman_action = None 
        if "pacman" in agents_directions:
            if pacman_action_index == 0:  # Right
                pacman_action = RIGHT
            elif pacman_action_index == 1:  # Down
                pacman_action = DOWN
            elif pacman_action_index == 2:  # Up
                pacman_action = UP
            elif pacman_action_index == 3:  # Left
                pacman_action = LEFT
            # else:
            #     print("Pacman actions provided" , agents_directions["pacman"])
            
        
            # print("Observation space for pacman:", env.observation_space("pacman"))
            # print("Action space for pacman:", env.action_space("pacman"))
            # print("Observation space for ghosts:", env.observation_space("ghosts"))
            # print("Action space for ghosts:", env.action_space("ghosts"))
            
            # Process actions for ghost
            # agents_directions["ghost"] = agents_directions["ghost"].squeeze(0).argmax(dim=-1).item()
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
                    "ghosts": [ghost_action, None , None , None ]
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
                # if distance == 0:
                #     distance = 1e-6  # Avoid division by 0

                # # Positive reward for being closer to Pacman
                # ghost_reward += (1 / distance)* 20  

                if distance < 1:
                    ghost_reward += 15  # Reward for catching Pacman
                    
            # Print the rewards for both agents
            # print(f"Pacman Reward: {pacman_reward}, Ghost Reward: {ghost_reward}")

            # Update terminated and truncated flags
            terminated = {a: self.game.done for a in self.agents}
            truncated = {a: False for a in self.agents}

            # Return observations and rewards
            observations = self._getobs()
            reward = {"pacman": pacman_reward, "ghost": ghost_reward}
            # print(reward)
            info = {a: {} for a in self.agents}
            
            step_reward = reward if reward != 0 else TIME_PENALITY

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



# Ensure plot directory exists
plot_dir = "./training_plots"
os.makedirs(plot_dir, exist_ok=True)

# Initialize environment and other components
if __name__ == "__main__":
    env_not_render = PacmanEnv(render_mode=None, mode=SCARY_1_MODE, move_mode=DISCRETE_STEPS_MODE, clock_tick=0, pacman_lives=3, maze_mode=MAZE1, pac_pos_mode=RANDOM_PAC_POS)
    env_render = PacmanEnv(render_mode="human", mode=SCARY_1_MODE, move_mode=DISCRETE_STEPS_MODE, clock_tick=10, pacman_lives=3, maze_mode=MAZE1, pac_pos_mode=RANDOM_PAC_POS)

    # model_path = "./models/maddpg"
    chkpt_dir = "./tmp/maddpg/"
    
    # os.makedirs(model_path, exist_ok=True)
    os.makedirs(chkpt_dir, exist_ok=True)

    env = env_not_render
    possible_agents = env.possible_agents 
    
    actor_dims = []
    for agent in env.possible_agents:
        if agent == "pacman":
            actor_dims.append(np.prod(env.observation_space(agent).shape))
        elif agent == "ghost":
            actor_dims.append(np.prod(env.observation_space(agent).shape))
            
    joint_state_dim = sum(actor_dims)
    joint_action_dim = sum([env.action_space(agent).n for agent in env.possible_agents])
    
    critic_dims = [joint_state_dim, joint_action_dim]
    # print("critic dims", critic_dims)
    
    n_actions = 5

    # Initialize MADDPG agents
    maddpg_agents = MADDPG(
        actor_dims,
        critic_dims,
        n_agents=len(env.possible_agents),
        n_actions=n_actions, 
        possible_agents=env.possible_agents,
        env=env,
        alpha=0.001,
        beta=0.01,
        gamma=0.99,
        tau=0.001,
        chkpt_dir=chkpt_dir,
        device='cuda',
    )

    memory = MultiAgentReplayBuffer(
        max_size=1000000,
        critic_dims=critic_dims,
        actor_dims=actor_dims,
        possible_agents=possible_agents,
        n_agents=len(env.possible_agents),
        n_actions=n_actions,
        batch_size=500,
        env=env
    )

    num_episodes = 50000
    MAX_STEPS = 1000
    n_agents = len(possible_agents)
    PRINT_INTERVAL =  5
    total_steps = 0
    best_score = -np.inf 
    score_history = []
    
    

    is_training = True
    evaluate = False
    
    
    episode_rewards = []
    pacman_rewards = []
    ghost_rewards = []

    actor_losses = []
    critic_losses = []
    
    episode_lengths = []
    def plot_training():
        plt.figure(figsize=(20, 10))

        # Total Episode Reward (Top-left)
        plt.subplot(2, 3, 1)
        plt.plot(episode_rewards, label="Total Reward")
        plt.xlabel("Episodes")
        plt.ylabel("Reward")
        plt.title("Total Episode Reward")
        plt.legend()

        # Agent-Specific Rewards (Top-middle)
        plt.subplot(2, 3, 2)
        plt.plot(pacman_rewards, label="Pacman Reward", color="blue")
        plt.plot(ghost_rewards, label="Ghosts Reward", color="orange")
        plt.xlabel("Episodes")
        plt.ylabel("Reward")
        plt.title("Agent-Specific Rewards")
        plt.legend()

        # Average Episode Length (Top-right)
        mean_episode_lengths = np.cumsum(episode_lengths) / np.arange(1, len(episode_lengths) + 1)
        plt.subplot(2, 3, 3)
        plt.plot(mean_episode_lengths, label="Average Episode Length", color="green")
        plt.xlabel("Episodes")
        plt.ylabel("Average Length")
        plt.title("Average Episode Length")
        plt.legend()

        # Actor Loss (Bottom-left)
        plt.subplot(2, 3, 4)
        plt.plot(actor_losses, label='Actor Loss')
        plt.xlabel('Episode')
        plt.ylabel('Loss')
        plt.title('Actor Loss over Time')
        plt.legend()

        # Critic Loss (Bottom-middle)
        plt.subplot(2, 3, 5)
        plt.plot(critic_losses, label='Critic Loss')
        plt.xlabel('Episode')
        plt.ylabel('Loss')
        plt.title('Critic Loss over Time')
        plt.legend()

        # Empty subplot for spacing (Bottom-right)
        plt.subplot(2, 3, 6)
        plt.axis('off')

        # Save the combined plot
        plot_filename = os.path.join(plot_dir, f"training_plot_episode_{episode}.png")
        plt.tight_layout()
        plt.savefig(plot_filename)
        plt.close()
        

    if is_training:
            print("Training new MADDPG model...")
            for episode in range(num_episodes):
                obs, _ = env.reset()
                score = 0
                done = [False] * n_agents
                episode_step = 0
                pacman_episode_reward = 0
                ghost_episode_reward = 0
                total_reward = 0
                episode_length = 0 
                
                
                while not any(done):
                    if evaluate:
                        env.render()

                    actions = {}
                    for agent in env.possible_agents:
                        raw_obs = [obs[agent] for agent in env.possible_agents]
                        actions = maddpg_agents.choose_action(raw_obs)
                        obs_, rewards, terminated, truncated, info = env.step(actions)

                    state = obs_list_to_state_vector([obs[agent] for agent in env.possible_agents])
                    state_ = obs_list_to_state_vector([obs_[agent] for agent in env.possible_agents])

                    if episode_step >= MAX_STEPS:
                        done = [True] * n_agents

                    total_reward += sum(rewards.values())
                    pacman_episode_reward += rewards["pacman"]
                    ghost_episode_reward += rewards["ghost"]

                    memory.store_transition(obs, state, actions, rewards, obs_, state_, done)

                    # Perform learning every step
                    maddpg_agents.learn(memory)

                    # Track losses for each agent after learning step
                    for agent in maddpg_agents.agents:
                        actor_losses.append(agent.actor_loss)  # Capture the actor loss
                        critic_losses.append(agent.critic_loss)  # Capture the critic loss

                    obs = obs_
                    score += sum(rewards.values())
                    episode_step += 1
                    episode_length += 1  # This is where the episode length is updated

                    total_steps += 1

                # After each episode ends, append the episode length to the list
                episode_lengths.append(episode_length)

                # Track rewards for each agent
                episode_rewards.append(total_reward)
                pacman_rewards.append(pacman_episode_reward)
                ghost_rewards.append(ghost_episode_reward)


                score_history.append(score)
                avg_score = np.mean(score_history[-100:])
                # print(f"Episode {episode}, Score: {score}, Average Score: {avg_score:.2f}, Total Reward: {total_reward:.2f}")

                # Save plot every PRINT_INTERVAL episodes
                if episode % PRINT_INTERVAL == 0 and episode > 0:
                    print(f"Episode {episode} - Plotting and saving rewards")
                    plot_training()  # Plot and save rewards

                # Save model checkpoint 
                maddpg_agents.save_checkpoint()

            # Save the final model after all episodes are complete
            print("Training Completed!")

