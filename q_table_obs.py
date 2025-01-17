from run import GameController
from constants import *
import numpy as np
import random
import pickle
from queue import PriorityQueue

# test_maze = np.array([
#     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
#     [1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1],
#     [1, 2, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 2, 1, 1, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 2, 1],
#     [1, 3, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 2, 1, 1, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 3, 1],
#     [1, 2, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 2, 1, 1, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 2, 1],
#     [1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1],
#     [1, 2, 1, 1, 1, 1, 2, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 2, 1, 1, 1, 1, 2, 1],
#     [1, 2, 1, 1, 1, 1, 2, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 2, 1, 1, 1, 1, 2, 1],
#     [1, 2, 2, 2, 2, 2, 2, 1, 1, 2, 2, 2, 2, 1, 1, 2, 2, 2, 2, 1, 1, 2, 2, 2, 2, 2, 2, 1],
#     [1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1],
#     [1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1],
#     [1, 1, 1, 1, 1, 1, 2, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 2, 1, 1, 1, 1, 1, 1],
#     [1, 1, 1, 1, 1, 1, 2, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 2, 1, 1, 1, 1, 1, 1],
#     [1, 1, 1, 1, 1, 1, 2, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 2, 1, 1, 1, 1, 1, 1],
#     [0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 1, 1, 1, 1, 1, 1, 6, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     [1, 1, 1, 1, 1, 1, 2, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 2, 1, 1, 1, 1, 1, 1],
#     [1, 1, 1, 1, 1, 1, 2, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 2, 1, 1, 1, 1, 1, 1],
#     [1, 1, 1, 1, 1, 1, 2, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 2, 1, 1, 1, 1, 1, 1],
#     [1, 1, 1, 1, 1, 1, 2, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 2, 1, 1, 1, 1, 1, 1],
#     [1, 1, 1, 1, 1, 1, 2, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 2, 1, 1, 1, 1, 1, 1],
#     [1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1],
#     [1, 2, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 2, 1, 1, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 2, 1],
#     [1, 2, 1, 1, 1, 1, 6, 1, 1, 1, 1, 1, 2, 1, 1, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 2, 1],
#     [1, 3, 2, 2, 1, 1, 5, 0, 0, 0, 0, 0, 0, 6, 0, 2, 2, 2, 2, 2, 2, 2, 1, 1, 2, 2, 3, 1],
#     [1, 1, 1, 2, 1, 1, 6, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 1],
#     [1, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 1],
#     [1, 2, 2, 2, 2, 2, 2, 1, 1, 2, 2, 2, 2, 1, 1, 2, 2, 2, 2, 1, 1, 2, 2, 2, 2, 2, 2, 1],
#     [1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1],
#     [1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1],
#     [1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1],
#     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
# ])


def manh_dist (tile1 , tile2):
    x1 , y1 = tile1
    x2 , y2 = tile2
    return abs(x1 - x2) + abs(y1 - y2)

def get_grid ():
    grid = []
    for row in range (GAME_ROWS):
        for col in range (GAME_COLS):
            grid.append((row , col))
    return grid

def get_direction(start , next_tile):
    direction = STOP
    if next_tile[0] == start[0] + 1:
        direction = DOWN
    elif next_tile[0] == start[0] - 1:
        direction = UP
    elif next_tile[1] == start[1] + 1:
        direction = RIGHT
    elif next_tile[1] == start[1] - 1:
        direction = LEFT
    return direction

def get_direction_idx (direction):
    if direction == UP:
        return 0
    elif direction == LEFT:
        return 1
    elif direction == DOWN:
        return 2
    elif direction == RIGHT:
        return 3
    elif direction == STOP:
        return 4

def get_direction_value (idx):
    if idx == 0:
        return UP
    elif idx == 1:
        return LEFT
    elif idx == 2:
        return DOWN
    elif idx == 3:
        return RIGHT
    elif idx == 4:
        return STOP

def a_star (maze , start , goal):
    if start == goal:
        return STOP , 0 , None

    start = (start[1] , start[0])
    goal= (goal[1] , goal[0])

    g_score = {cell : float("inf") for cell in get_grid()}
    g_score [start] = 0
    f_score = {cell : float("inf") for cell in get_grid()}
    f_score [start] = manh_dist(start , goal)

    open = PriorityQueue()
    open.put((f_score [start] , manh_dist(start , goal) , start))
    child_cell = (0,0)
    path = {}
    path_found = False

    while not open.empty():
        curr_cell = open.get()[2]
        if curr_cell == goal:
            path_found = True
            break
        wall = True
        for d in "ESNW":
            if d == "E":
                if curr_cell[1] < GAME_COLS - 1 and maze[curr_cell[0]][curr_cell[1] + 1] != WALL_MAZE:
                    child_cell = (curr_cell[0] , curr_cell[1] + 1)
                    wall = False
            elif d == "W":
                if curr_cell[1] > 0 and maze[curr_cell[0]][curr_cell[1] - 1] != WALL_MAZE:
                    child_cell = (curr_cell[0] , curr_cell[1] - 1)
                    wall = False
            elif d == "N":
                if curr_cell[0] > 0 and maze[curr_cell[0] - 1][curr_cell[1]] != WALL_MAZE:
                    child_cell = (curr_cell[0] - 1 , curr_cell[1])
                    wall = False
            elif d == "S":
                if curr_cell[0] < GAME_ROWS and maze[curr_cell[0] + 1][curr_cell[1]] != WALL_MAZE:
                    child_cell = (curr_cell[0] + 1 , curr_cell[1])
                    wall = False

            if not wall:
                temp_g_score = g_score[curr_cell] + 1
                temp_f_score = temp_g_score + manh_dist(child_cell , goal)

                if temp_f_score < f_score[child_cell]:
                    path[child_cell] = curr_cell
                    g_score[child_cell] = temp_g_score
                    f_score[child_cell] = temp_f_score
                    open.put((f_score[child_cell] , manh_dist(child_cell , goal) , child_cell))            
    if path_found:
        fwd_path = {}
        cell = goal
        while cell != start:
            fwd_path[path[cell]] = cell
            cell = path[cell]
        path_length = len(fwd_path)  
        next_tile = fwd_path[start]
        direction = STOP
        if next_tile[0] == start[0] + 1:
            direction = DOWN
        elif next_tile[0] == start[0] - 1:
            direction = UP
        elif next_tile[1] == start[1] + 1:
            direction = RIGHT
        elif next_tile[1] == start[1] - 1:
            direction = LEFT
        else:
            raise Exception("an error happened in the A star (child of the start node isn't next to it)")

        return direction , path_length , fwd_path
    else:
         return None , None , None

def all_astar_paths (maze , start , goal):
    dirs = []
    ########################### if goal is exactly next to start
    if start[0] + 1 == goal[0] and start[1] == goal[1]: #goal is exactly on the right
        direction , path_length , path = a_star (maze , start , goal)
        dirs.append((path_length , direction))
    if start[0] - 1 == goal[0] and start[1] == goal[1]: #goal is exactly on the left 
        direction , path_length , path = a_star (maze , start , goal)
        dirs.append((path_length , direction))
    if start[1] + 1 == goal[1] and start[0] == goal[0]: #goal is exactly don
        direction , path_length , path = a_star (maze , start , goal)
        dirs.append((path_length , direction))
    if start[1] - 1 == goal[1] and start[0] == goal[0]: #goal is exactly above
        direction , path_length , path = a_star (maze , start , goal)
        dirs.append((path_length , direction))

    ########################### ELSE: the goal is aay from the start
    if start[0] < GAME_COLS - 1 and maze[start[1]][start[0] + 1] != WALL_MAZE:  #tile on right 
        tile_on_right = (start[0] + 1, start[1]) 
        direction , path_length , path = a_star (maze , tile_on_right , goal)
        if direction != None and direction != STOP and path_length != None and path_length !=0 :
            if path[(tile_on_right[1] , tile_on_right[0])] != (start[1] , start[0]):
                path_length += 1
                dirs.append((path_length , RIGHT))

    if start[0] > 0 and maze[start[1]][start[0] - 1] != WALL_MAZE:   #tile on left
        tile_on_left = (start[0] - 1 , start[1])
        direction , path_length , path = a_star (maze , tile_on_left , goal)
        if direction != None and direction != STOP and path_length != None and path_length !=0 :
            if path[(tile_on_left[1] , tile_on_left[0])] != (start[1] , start[0]):
                path_length += 1
                dirs.append((path_length , LEFT))
    if start[1] < GAME_ROWS - 1 and maze[start[1] + 1][start[0]] != WALL_MAZE:  #tile don
        tile_down = (start[0] , start[1] + 1)
        direction , path_length , path = a_star (maze , tile_down , goal)
        if direction != None and direction != STOP and path_length != None and path_length !=0 :
            if path[(tile_down[1] , tile_down[0])] != (start[1] , start[0]):
                path_length += 1
                dirs.append((path_length , DOWN))
    if start[1] > 0 and maze[start[1] - 1][start[0]] != WALL_MAZE:  #tile up
        tile_up = (start[0] , start[1] - 1)
        direction , path_length , path = a_star (maze , tile_up , goal)
        if direction != None and direction != STOP and path_length != None and path_length !=0 :
            if path[(tile_up[1] , tile_up[0])] != (start[1] , start[0]):
                path_length += 1
                dirs.append((path_length , UP))
    return dirs


def check_trapped (game , walls_obs , danger_paths):
    exit_path = [float("-inf") , STOP]  #first idx represents the difference beteen pacman and closest ghost when reaching the exit and the second represents the direction to reach this exit
    trapped = True

    if walls_obs[0] != 1:   # there is no wall on north
        path_length_to_exit = 0
        current_tile = game.pacman.tile
        while True:
            if current_tile[1] <= 0:
                break
            else:
                tile_up = (current_tile[0] , current_tile[1] - 1)
                if game.maze_map[tile_up[1]][tile_up[0]] == GCC_MAZE:
                    break
                path_length_to_exit += 1
                if tile_up[0] < GAME_COLS - 1:
                    tile_right = (tile_up[0]+1 , tile_up[1])
                    if game.maze_map[tile_right[1]][tile_right[0]] != WALL_MAZE and game.maze_map[tile_right[1]][tile_right[0]] != GCC_MAZE and game.maze_map[tile_right[1]][tile_right[0]] != GCF_MAZE:
                        trapped = False
                        if (danger_paths[UP] -  path_length_to_exit) - path_length_to_exit > exit_path[0]:  #get biggest difference in distance between me an closer exit and the ghost and closer exit 
                            exit_path[0] = (danger_paths[UP] -  path_length_to_exit) - path_length_to_exit
                            exit_path[1] = UP
                        break
                if tile_up[0] > 0:
                    tile_left = (tile_up[0] - 1 , tile_up[1])
                    if game.maze_map[tile_left[1]][tile_left[0]] != WALL_MAZE and game.maze_map[tile_left[1]][tile_left[0]] != GCC_MAZE and game.maze_map[tile_left[1]][tile_left[0]] != GCF_MAZE:
                        trapped = False
                        if (danger_paths[UP] -  path_length_to_exit) - path_length_to_exit > exit_path[0]:  #get biggest difference in distance between me an closer exit and the ghost and closer exit 
                            exit_path[0] = (danger_paths[UP] -  path_length_to_exit) - path_length_to_exit
                            exit_path[1] = UP
                        break
            current_tile = tile_up

    if walls_obs[1] != 1:   # there is no wall on west
        path_length_to_exit = 0
        current_tile = game.pacman.tile
        while True:
            if current_tile[0] <= 0:
                break
            else:
                tile_left = (current_tile[0] - 1 , current_tile[1])
                if game.maze_map[tile_left[1]][tile_left[0]] == GCC_MAZE:
                    break
                path_length_to_exit += 1
                if tile_left[1] > 0 :
                    tile_up = (tile_left[0] , tile_left[1] - 1)
                    if game.maze_map[tile_up[1]][tile_up[0]] != WALL_MAZE and game.maze_map[tile_up[1]][tile_up[0]] != GCC_MAZE and game.maze_map[tile_up[1]][tile_up[0]] != GCF_MAZE:
                        trapped = False
                        if (danger_paths[LEFT] -  path_length_to_exit) - path_length_to_exit > exit_path[0]:  #get biggest difference in distance between pacman an closer exit and the ghost and closer exit 
                            exit_path[0] = (danger_paths[LEFT] -  path_length_to_exit) - path_length_to_exit
                            exit_path[1] = LEFT
                        break
                if tile_left[1] < GAME_ROWS - 1:
                    tile_down = (tile_left[0] , tile_left[1] + 1)
                    if game.maze_map[tile_down[1]][tile_down[0]] != WALL_MAZE and game.maze_map[tile_down[1]][tile_down[0]] != GCC_MAZE and game.maze_map[tile_down[1]][tile_down[0]] != GCF_MAZE:
                        trapped = False
                        if (danger_paths[LEFT] -  path_length_to_exit) - path_length_to_exit > exit_path[0]:  #get biggest difference in distance between pacman an closer exit and the ghost and closer exit 
                            exit_path[0] = (danger_paths[LEFT] -  path_length_to_exit) - path_length_to_exit
                            exit_path[1] = LEFT
                        break        
            current_tile = tile_left

    if walls_obs[2] != 1:   # there is no wall south
        path_length_to_exit = 0
        current_tile = game.pacman.tile
        while True:
            if current_tile[1] >= GAME_ROWS - 1:
                break
            else:
                tile_down = (current_tile[0] , current_tile[1] + 1)
                if game.maze_map[tile_down[1]][tile_down[0]] == GCC_MAZE:
                    break
                path_length_to_exit += 1
                if tile_down[0] < GAME_COLS - 1:
                    tile_right = (tile_down[0]+1 , tile_down[1])
                    if game.maze_map[tile_right[1]][tile_right[0]] != WALL_MAZE and game.maze_map[tile_right[1]][tile_right[0]] != GCC_MAZE and game.maze_map[tile_right[1]][tile_right[0]] != GCF_MAZE:
                        trapped = False
                        if (danger_paths[DOWN] -  path_length_to_exit) - path_length_to_exit > exit_path[0]:  #get biggest difference in distance between me an closer exit and the ghost and closer exit 
                            exit_path[0] = (danger_paths[DOWN] -  path_length_to_exit) - path_length_to_exit
                            exit_path[1] = DOWN
                        break
                if tile_down[0] > 0:
                    tile_left = (tile_down[0] - 1 , tile_down[1])
                    if game.maze_map[tile_left[1]][tile_left[0]] != WALL_MAZE and game.maze_map[tile_left[1]][tile_left[0]] != GCC_MAZE and game.maze_map[tile_left[1]][tile_left[0]] != GCF_MAZE:
                        trapped = False
                        if (danger_paths[DOWN] -  path_length_to_exit) - path_length_to_exit > exit_path[0]:  #get biggest difference in distance between me an closer exit and the ghost and closer exit 
                            exit_path[0] = (danger_paths[DOWN] -  path_length_to_exit) - path_length_to_exit
                            exit_path[1] = DOWN
                        break        
            current_tile = tile_down
    
    if walls_obs[3] != 1:   # there is no wall on east
        path_length_to_exit = 0
        current_tile = game.pacman.tile
        while True:
            if current_tile[0] >= GAME_COLS - 1:
                break
            else:
                tile_right = (current_tile[0] + 1 , current_tile[1])
                if game.maze_map[tile_right[1]][tile_right[0]] == GCC_MAZE:
                    break
                path_length_to_exit += 1
                if tile_right[1] > 0 :
                    tile_up = (tile_right[0] , tile_right[1] - 1)
                    if game.maze_map[tile_up[1]][tile_up[0]] != WALL_MAZE and game.maze_map[tile_up[1]][tile_up[0]] != GCC_MAZE and game.maze_map[tile_up[1]][tile_up[0]] != GCF_MAZE:
                        trapped = False
                        if (danger_paths[RIGHT] -  path_length_to_exit) - path_length_to_exit > exit_path[0]:  #get biggest difference in distance between me an closer exit and the ghost and closer exit 
                            exit_path[0] = (danger_paths[RIGHT] -  path_length_to_exit) - path_length_to_exit
                            exit_path[1] = RIGHT
                        break
                if tile_right[1] < GAME_ROWS - 1:
                    tile_down = (tile_right[0] , tile_right[1] + 1)
                    if game.maze_map[tile_down[1]][tile_down[0]] != WALL_MAZE and game.maze_map[tile_down[1]][tile_down[0]] != GCC_MAZE and game.maze_map[tile_down[1]][tile_down[0]] != GCF_MAZE:
                        trapped = False
                        if (danger_paths[RIGHT] -  path_length_to_exit) - path_length_to_exit > exit_path[0]:  #get biggest difference in distance between me an closer exit and the ghost and closer exit 
                            exit_path[0] = (danger_paths[RIGHT] -  path_length_to_exit) - path_length_to_exit
                            exit_path[1] = RIGHT
                        break        
            current_tile = tile_right
    if exit_path[0] <= 0:
        trapped = True
    return (trapped , exit_path)


def get_observation (game):
    observation = np.zeros(10 , dtype=int)
    ## get Walls positions
    pacman_tile = game.pacman.tile
    maze_map = game.maze_map

    if pacman_tile[0] < GAME_COLS - 1 and maze_map[pacman_tile[1]][pacman_tile[0] + 1] == WALL_MAZE:  #all on the right
        observation[3] = 1
    if pacman_tile[0] > 0 and maze_map[pacman_tile[1]][pacman_tile[0] - 1] == WALL_MAZE:  #all on the left
        observation[1] = 1
    if pacman_tile[1] < GAME_ROWS - 1 and maze_map[pacman_tile[1] + 1][pacman_tile[0]] == WALL_MAZE:  #all don
        observation[2] = 1
    if pacman_tile[1] > 0 and maze_map[pacman_tile[1] - 1][pacman_tile[0]] == WALL_MAZE:  #all in the north
        observation[0] = 1

    ## get ghosts directions
    ghost_dirs = np.zeros(4 , dtype=int)
    ghost_paths = []
    for ghost in game.ghosts:
        if ghost.mode.current is CHASE or ghost.mode.current is SCATTER:
            if maze_map[ghost.tile[1]][ghost.tile[0]] == GCC_MAZE:  # e found a chasing ghost :)
                paths_to_ghost = all_astar_paths(maze_map , game.pacman.tile , ghost.tile)
                for path in paths_to_ghost:
                    if path[0] <= 8:
                        ghost_paths.append((path[0] , path[1]))
                        ghost_idx = get_direction_idx(path[1])
                        ghost_dirs[ghost_idx] = 1
    observation [5:9] = ghost_dirs


    ## check if all the roads are blocked
    danger_paths = {UP : float("inf") , DOWN: float("inf") , RIGHT: float("inf") , LEFT: float("inf")}
    for p in ghost_paths:
        if p[0] < danger_paths[p[1]]:
            danger_paths[p[1]] = p[0]

    prefered_dir = STOP

    ghosts_walls = np.bitwise_or(ghost_dirs, observation[:4])
    all_ones = np.all(ghosts_walls == 1) 
    trapped = False 
    direction_closest_pellet = [float("inf") , STOP]  #first idx is for the distance and second idx is the direction
    direction_closest_freight_ghost = [5 , STOP]
    direction_closest_pellet_escape_ghosts = [float("inf") , STOP]
    if all_ones:            #if there is no free path
        trapped , exit_path = check_trapped(game , observation[:4] , danger_paths) 

        if not trapped:
            prefered_dir = exit_path[1]
            #print("prefered_direction_escape: ",prefered_dir)
        else:
            #in this case we are trapped and we will choose the direction to the closest pellet
            for pellet in game.pellets.pelletList:
                direction_pellet , path_length_pellet , _ = a_star(maze_map , game.pacman.tile , pellet.tile)
                if path_length_pellet < direction_closest_pellet[0]: 
                    direction_closest_pellet[0] = path_length_pellet
                    direction_closest_pellet[1] = direction_pellet
                    # print(direction_pellet)
                    # print(pellet.tile)
            prefered_dir =  direction_closest_pellet[1]
            #print("prefered_dir_closest_pellet_لا فائدة: ",prefered_dir)
    observation[9] = trapped

    if prefered_dir == STOP:  # there is an open path for the pacman to move in and this path doesn't contain any wall or any ghost with distance <= 8
        ## get the direction to the closest freight ghost
        for ghost in game.ghosts:
            if ghost.mode.current is FREIGHT:  #We found a frieght ghost
                path_freight_ghost = all_astar_paths(game.maze_map ,game.pacman.tile , ghost.tile)
                for path in path_freight_ghost:
                    if danger_paths[path[1]] == float("inf"):  #if this path doesn't contain a ghost
                        if path[0] <= direction_closest_freight_ghost[0]:
                            direction_closest_freight_ghost[0] = path[0]
                            direction_closest_freight_ghost[1] = path[1]
        prefered_dir = direction_closest_freight_ghost[1]
        #print("prefered_dir_freight_ghost: ",prefered_dir)
        
    # if prefered_dir == STOP:  # we didn't found a direction to a close freight ghost then no ge the direction to the closest pellet
    #     for pellet in game.pellets.pelletList:
    #         paths_to_pellets = all_astar_paths(game.maze_map , game.pacman.tile , pellet.tile)
    #         for path in paths_to_pellets:
    #             if danger_paths[path[1]] == float("inf"): #if this path doesn't contain a ghost
    #                 if path[0] < direction_closest_pellet_escape_ghosts[0]:
    #                     direction_closest_pellet_escape_ghosts[0] = path[0]
    #                     direction_closest_pellet_escape_ghosts[1] = path[1]
    #     prefered_dir = direction_closest_pellet_escape_ghosts[1]
    #     print("prefered_direction_closest_pellet: ",prefered_dir)
    if prefered_dir == STOP:  # we didn't find a direction to a close freight ghost then now get the direction to the closest pellet
        for pellet in game.pellets.pelletList:
            direction_pellet , path_length_pellet , _ = a_star(game.maze_map , game.pacman.tile , pellet.tile)
            if danger_paths[direction_pellet] == float("inf"): #if this path doesn't contain a ghost
                if path_length_pellet < direction_closest_pellet_escape_ghosts[0]:
                    direction_closest_pellet_escape_ghosts[0] = path_length_pellet
                    direction_closest_pellet_escape_ghosts[1] = direction_pellet
        prefered_dir = direction_closest_pellet_escape_ghosts[1]
        #print("prefered_direction_closest_pellet: ",prefered_dir)

    prefered_dir = get_direction_idx(prefered_dir)
    observation[4] = prefered_dir
    
    return observation

# game = GameController(rlTraining=True , mode = SCARY_2_MODE , move_mode = DISCRETE_STEPS_MODE , clock_tick= 10 , pacman_lives=1 , maze_mode=MAZE1 , pac_pos_mode=NORMAL_PAC_POS)

# direc , path_length , path = a_star(game.maze_map , game.pacman.tile , (1,1))

# start = (23, 13)
# #agent_direction = get_direction(start , path[start])
# done = False
# dire = LEFT
# while not done:
#     observation = get_observation(game)
#     agent_direction = observation[4]
#     agent_direction = get_direction_value(agent_direction)
#     # print("agent_direction" , agent_direction)
#     # print("pac tile: " , game.pacman.tile)
#     print(observation)
#     game.update(render=False ,agent_direction = agent_direction)
#     # start = path[start]
#     # agent_direction = get_direction(start , path[start])
#     # g = {4 : "red: " , 5 : "pink: "}
#     # for ghost in game.ghosts:
#     #     paths = all_astar_paths(game.maze_map , game.pacman.tile , ghost.tile)
#     #     print(g[ghost.name])
#     #     print(paths)
#     #     print(ghost.tile)
#     #     print("**")
#     # print(game.pacman.tile)
#     # print(game.done)
#     # print("*********************")

#     ########### test maze
#     # game.maze_map = test_maze
#     # game.pacman.tile = (6,23)
#     # game.ghosts[0].tile = (13,23)
#     # game.ghosts[1].tile = (6,22)
#     # game.ghosts[3].tile = (6,24)
    
#     print("*****************************")
#     done = game.done

#     #print(game.done)

