import pygame
from vector import Vector2
from constants import *
import numpy as np


class Pellet(object):
    def __init__(self, row, column):
        self.name = PELLET
        self.position = Vector2(column * TILEWIDTH, row * TILEHEIGHT)
        self.tile = (int((self.position.x // TILEWIDTH) - 1), int((self.position.y // TILEHEIGHT) - 4))

        self.color = WHITE
        self.radius = int(2 * TILEWIDTH / 16)
        self.collideRadius = int(2 * TILEWIDTH / 16)
        self.points = 10
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
        self.points = 50
        self.flashTime = 0.2
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
        self.set_maze_pellets_map()
        self.createPelletList(pelletfile)

    def set_maze_pellets_map (self):
        self.map_init_pell_rewards = np.empty((GAME_ROWS,GAME_COLS), dtype=object)
        for row in range (GAME_ROWS):
            for col in range (GAME_COLS):
                self.map_init_pell_rewards[row][col] = np.array([0,0]) #first index is for rewards in this tile ,  second is for the ghost penality in the tile
        

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
                    self.map_init_pell_rewards[pel.tile[1]][pel.tile[0]][0] = pel.points
                    ###
                elif data[row][col] in ["P", "p"]:
                    pp = PowerPellet(row, col)
                    self.pelletList.append(pp)
                    self.powerpellets.append(pp)

                    ### put the power pellets reward in the maze
                    self.map_init_pell_rewards[pp.tile[1]][pp.tile[0]][0] = pp.points
                    ###
    def readPelletfile(self, pelletfile):
        return np.loadtxt(pelletfile, dtype="<U1")

    def isEmpty(self):
        if len(self.pelletList) == 0:
            return True
        return False

    def render(self, screen):
        for pellet in self.pelletList:
            pellet.render(screen)
