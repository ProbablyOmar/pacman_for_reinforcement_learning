import pygame
from vector import Vector2
from constants import *
import numpy as np


class Pellet(object):
    def __init__(self, row, column):
        self.name = PELLET
        self.position = Vector2(column * TILEWIDTH, row * TILEHEIGHT)
        self.tile = (int((self.position.x // TILEWIDTH)), int((self.position.y // TILEHEIGHT) - 3))

        self.color = WHITE
        self.radius = int(2 * TILEWIDTH / 16)
        self.collideRadius = int(2 * TILEWIDTH / 16)
        self.points = PELLET_REWARD
        self.visible = True

    def render(self, screen):
        if self.visible:
            adjust = Vector2(TILEWIDTH, TILEHEIGHT) / 2
            p = self.position + adjust
            pygame.draw.circle(screen, self.color, p.asInt(), self.radius)


class PowerPellet(Pellet):
    def __init__(self, row, column):
        Pellet.__init__(self, row, column)
        self.name = POWERPELLET
        self.radius = int(8 * TILEWIDTH / 16)
        self.points = POWERPELLET_REWARD
        self.flashTime = 0.2   #0.2
        self.timer = 0

    def update(self, dt):
        self.timer += dt
        if self.timer >= self.flashTime:
            self.visible = not self.visible
            self.timer = 0


class PelletGroup(object):
    def __init__(self, pelletfile) -> None:
        self.pelletList = []
        self.powerpellets = []
        self.numEaten = 0
        self.map_init_pell_rewards = np.zeros((GAME_ROWS,GAME_COLS), dtype=int)
        self.createPelletList(pelletfile)

    def updatePoints(self):
        for pellet in self.pelletList:
            pellet.points += PELLET_REWARD_UPDATE

    def update(self, dt):
        for powerpellet in self.powerpellets:
            powerpellet.update(dt)

    def createPelletList(self, pelletfile):
        data = self.readPelletfile(pelletfile)
        for row in list(range(data.shape[0])):
            for col in list(range(data.shape[1])):
                if data[row][col] in [".", "+"]:
                    pel = Pellet(row, col)
                    self.pelletList.append(pel)
                    ### put the pellets reward in the maze
                    self.map_init_pell_rewards[pel.tile[1]][pel.tile[0]]= PELLET_MAZE
                    ###
                elif data[row][col] in ["P", "p"]:
                    pp = PowerPellet(row, col)
                    self.pelletList.append(pp)
                    self.powerpellets.append(pp)

                    ### put the power pellets reward in the maze
                    self.map_init_pell_rewards[pp.tile[1]][pp.tile[0]] = PP_MAZE
                    ###

                ## if its a wall in the maze
                elif (data[row][col] in ["X" , "="] or (data[row][col]).isdigit())  and row >= 3 and row <= 33:
                    self.map_init_pell_rewards[row-3][col] = WALL_MAZE

    def readPelletfile(self, pelletfile):
        return np.loadtxt(pelletfile, dtype="<U1")

    def isEmpty(self):
        if len(self.pelletList) == 0:
            return True
        return False

    def render(self, screen):
        for pellet in self.pelletList:
            pellet.render(screen)

# mazedata = MazeData()
# mazedata.loadMaze(0)
# mazeFolderPath = Path("./mazes") / (mazedata.obj.name)
# mazeFilePath = mazeFolderPath / (mazedata.obj.name + ".txt")
# pellets = PelletGroup(mazeFilePath.resolve())


# print(pellets.map_init_pell_rewards)
