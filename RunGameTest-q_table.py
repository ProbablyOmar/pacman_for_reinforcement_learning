from run import GameController
from constants import *
import numpy as np
import pickle
from queue import PriorityQueue
from q_table_obs import *
import os
import matplotlib.pyplot as plt
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

game = GameController(
    rlTraining=True,
    mode=NORMAL_MODE,
    move_mode=DISCRETE_STEPS_MODE,
    clock_tick=10,
    pacman_lives=3,
    maze_mode=MAZE1,
    pac_pos_mode=NORMAL_PAC_POS
)

q_table_path = "q_tables_sarsa_ghosts_complete/q_table_420_episodes.pkl"
if os.path.exists(q_table_path):
    with open(q_table_path, "rb") as file:
        q_table = pickle.load(file)
        for key in q_table:
            q_table[key] = q_table[key].to(device)
else:
    print(f"Error: Q-table file '{q_table_path}' not found.")
    exit()

done = False

while not done:
    observation = get_observation(game)
    observation_tensor = torch.tensor(observation, device=device)
    agent_direction = torch.argmax(q_table[tuple(observation)])
    agent_direction = get_direction_value(agent_direction)
    game.update(render=True, agent_direction=agent_direction)
    done = game.done
