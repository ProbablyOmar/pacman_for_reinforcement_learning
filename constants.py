TILEWIDTH = 16
TILEHEIGHT = 16
NROWS = 36
NCOLS = 28

GAME_ROWS = 31
GAME_COLS = 28

SCREENWIDTH = NCOLS * TILEWIDTH
SCREENHEIGHT = NROWS * TILEHEIGHT
SCREENSIZE = (SCREENWIDTH, SCREENHEIGHT)
BLACK = (0, 0, 0)
YELLOW = (255, 255, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
PINK = (255, 100, 150)
TEAL = (100, 255, 255)
ORANGE = (230, 190, 40)
GREEN = (0, 255, 0)

# NUMBER OF LEVELS
NUMBEROFLEVELS = 1

# DIRECTIONS
STOP = 0
UP = 1
DOWN = -1
LEFT = 2
RIGHT = -2
PORTAL = 3

# ENTITIES
NUMGHOSTS = 4
PACMAN = 0
PELLET = 1
POWERPELLET = 2
GHOST = 3
BLINKY = 4
PINKY = 5
INKY = 6
CLYDE = 7
FRUIT = 8


# MODES

SCATTER = 0
CHASE = 1
FREIGHT = 2
SPAWN = 3

# TEXT

SCORETXT = 0
LEVELTXT = 1
READYTXT = 2
PAUSETXT = 3
GAMEOVERTXT = 4

# Rewards
GHOST_PENALITY = -20
TIME_PENALITY = 0  ### -.01
GHOST_REWARD = 7.5

HIT_WALL_PENALITY = -3.5 
PELLET_REWARD = 10
POWERPELLET_REWARD = 50
FRUIT_REWARD = 100
FINISH_LEVEL_REWARD = 50

GHOST_UPDATE_REWARD = 0.5
GHOST_UPDATE_PENALITY = 2
GHOST_REWARDS = {GHOST_PENALITY : 1 , GHOST_REWARD : 2 , GHOST_REWARD+GHOST_UPDATE_REWARD : 3 , GHOST_REWARD+2*GHOST_UPDATE_REWARD : 4 ,  GHOST_REWARD+3*GHOST_UPDATE_REWARD : 5}


# GHOST_PENALITY = -20
# TIME_PENALITY = -0.2  ### -.01
# HIT_WALL_PENALITY = -3.5  
# GHOST_REWARD = 7.5
# PELLET_REWARD = 4
# POWERPELLET_REWARD = 10 
# FRUIT_REWARD = 5
# FINISH_LEVEL_REWARD = 50

# GHOST_UPDATE_REWARD = 0.5
# GHOST_UPDATE_PENALITY = 2
# GHOST_REWARDS = {GHOST_PENALITY : 1 , GHOST_REWARD : 2 , GHOST_REWARD+GHOST_UPDATE_REWARD : 3 , GHOST_REWARD+2*GHOST_UPDATE_REWARD : 4 ,  GHOST_REWARD+3*GHOST_UPDATE_REWARD : 5}




# GHOST_PENALITY = -50
# TIME_PENALITY = -1
# HIT_WALL_PENALITY = -3   
# GHOST_REWARD = 200
# PELLET_REWARD = 10
# POWERPELLET_REWARD = 50
# FRUIT_REWARD = 100
# FINISH_LEVEL_REWARD = 400
## recommendation: [−5,−2.5,0,2.5,5,7.5,10]
#GHOST_REWARDS = {-50 : 1 , 200 : 2 , 400 : 3 , 800 : 4 , 1600 : 5}
#

## maze map encodings
WALL_MAZE = 1
PELLET_MAZE = 2
PP_MAZE = 3
FRUIT_MAZE = 4

### game mode
SAFE_MODE = 0
NORMAL_MODE = 1

###

MAX_USELESS_STEPS = 50
#MAX_USELESS_STEPS = 70


