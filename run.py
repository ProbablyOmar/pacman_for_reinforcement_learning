import pygame
from pathlib import Path
from pygame.locals import *
from constants import *
from pacman import Pacman
from nodes import NodeGroup
from pellets import PelletGroup
from ghosts import GhostGroup
from fruit import Fruit
from pauser import Pauser
from text import TextGroup
from sprites import LifeSprites
from sprites import MazeSprites
from mazedata import MazeData
import numpy as np
import time
from copy import deepcopy

class GameController(object):
    def __init__(self, rlTraining=False):
        pygame.init()
        self.rlTraining = rlTraining
        self.screen = pygame.display.set_mode(SCREENSIZE, 0, 32)
        pygame.display.set_caption("Pacman")
        self.background = None
        self.background_norm = None
        self.background_flash = None
        self.clock = pygame.time.Clock()
        self.fruit = None
        self.pause = Pauser(not self.rlTraining)
        self.level = 0
        self.lives = 5
        self.score = 0
        self.RLreward = 0
        self.textgroup = TextGroup(rlTraining=self.rlTraining)
        self.lifesprites = LifeSprites(self.lives)
        self.flashBG = False
        self.flashTime = 0.2
        self.flashTimer = 0
        self.fruitcaptured = []
        self.mazedata = MazeData()
        self.gameOver = False
        self.startGame()
        


    def restartGame(self):
        self.gameOver = False
        self.lives = 5
        self.level = 0
        self.pause.paused = not self.rlTraining
        self.fruit = None
        self.startGame()
        self.score = 0
        self.textgroup.updateScore(self.score)
        self.textgroup.updateLevel(self.level)
        if not (self.rlTraining):
            self.textgroup.showText(READYTXT)
        self.lifesprites.resetLives(self.lives)
        self.fruitcaptured = []

    def resetLevel(self):
        self.gameOver = False
        self.pause.paused = not self.rlTraining
        self.pacman.reset()
        self.ghosts.reset()
        self.fruit = None
        if not (self.rlTraining):
            self.textgroup.showText(READYTXT)

    def nextLevel(self):
        self.showEntities()
        self.level += 1
        if self.level > NUMBEROFLEVELS:
            self.gameOver = True
        self.pause.pasued = True
        self.startGame()
        self.textgroup.updateLevel(self.level)

    def setBackground(self):
        self.background_norm = pygame.surface.Surface(SCREENSIZE).convert()
        self.background_norm.fill(BLACK)
        self.background_flash = pygame.surface.Surface(SCREENSIZE).convert()
        self.background_flash.fill(BLACK)
        self.background_norm = self.mazesprites.constructBackground(
            self.background_norm, self.level % 5
        )
        self.background_flash = self.mazesprites.constructBackground(
            self.background_flash, 5
        )
        self.flashBG = False
        self.background = self.background_norm

    def startGame(self):
        self.mazedata.loadMaze(self.level)
        mazeFolderPath = Path("./mazes") / (self.mazedata.obj.name)
        mazeFilePath = mazeFolderPath / (self.mazedata.obj.name + ".txt")
        mazeRotFilePath = mazeFolderPath / (self.mazedata.obj.name + "_rotation.txt")
        self.mazesprites = MazeSprites(
            mazeFilePath.resolve(), mazeRotFilePath.resolve()
        )
        self.setBackground()
        self.nodes = NodeGroup(mazeFilePath.resolve())
        self.mazedata.obj.setPortalPairs(self.nodes)
        self.mazedata.obj.connectHomeNodes(self.nodes)
        self.pacman = Pacman(
            self.nodes.getNodeFromTiles(*self.mazedata.obj.pacmanStart)
        )
        self.pellets = PelletGroup(mazeFilePath.resolve())
        self.ghosts = GhostGroup(self.nodes.getStartTempNode(), self.pacman)
        self.ghosts.pinky.setStartNode(
            self.nodes.getNodeFromTiles(*self.mazedata.obj.addOffset(2, 3))
        )
        self.ghosts.inky.setStartNode(
            self.nodes.getNodeFromTiles(*self.mazedata.obj.addOffset(0, 3))
        )
        self.ghosts.clyde.setStartNode(
            self.nodes.getNodeFromTiles(*self.mazedata.obj.addOffset(4, 3))
        )
        self.ghosts.setSpawnNode(
            self.nodes.getNodeFromTiles(*self.mazedata.obj.addOffset(2, 3))
        )
        self.ghosts.blinky.setStartNode(
            self.nodes.getNodeFromTiles(*self.mazedata.obj.addOffset(2, 0))
        )
        self.nodes.denyHomeAccess(self.pacman)
        self.nodes.denyHomeAccessList(self.ghosts)
        self.ghosts.inky.startNode.denyAccess(RIGHT, self.ghosts.inky)
        self.ghosts.clyde.startNode.denyAccess(LEFT, self.ghosts.clyde)
        self.mazedata.obj.denyGhostsAccess(self.ghosts, self.nodes)

        self.set_maze_map = deepcopy(self.pellets.map_init_pell_rewards)
        self.maze_map = deepcopy(self.set_maze_map)
        self.get_obs()

    def update(self, agent_direction=None, render=True, clocktick=60):
        self.RLreward = 0
        dt = self.clock.tick(clocktick) / 1000.0
        self.textgroup.update(dt)
        self.pellets.update(dt)
        if not self.pause.paused or self.rlTraining == True:
            self.ghosts.update(dt)
            if self.fruit is not None:
                self.fruit.update(dt)
            self.checkPelletEvents()

            #initialize the maze map before updating it
            self.maze_map = deepcopy(self.set_maze_map)

            self.checkGhostEvents()
            self.checkFruitEvents()

        if self.pacman.alive:
            if not self.pause.paused:
                self.pacman.update(dt, agent_direction)
        else:
            self.pacman.update(dt, agent_direction)

        if self.flashBG:
            self.flashTimer += dt
            if self.flashTimer >= self.flashTime:
                self.flashTimer = 0
                if self.background == self.background_norm:
                    self.background = self.background_flash
                else:
                    self.background = self.background_norm

        ################################################
        self.get_obs()

        afterPauseMethod = self.pause.update(dt)
        if afterPauseMethod is not None:
            afterPauseMethod()
        self.checkEvents()
        if render:
            self.render()

    def get_obs (self):
        ################################################
        #collecting walls directions
        walls_position = np.array([1,1,1,1] , dtype = np.bool_)
        for key in [UP , LEFT]:
            if self.pacman.validDirection(key):
                walls_position[key+1] = 0
        for key in [RIGHT , DOWN]:
            if self.pacman.validDirection(key):
                walls_position[key+2] = 0

        #### best direction of pacman and ghosts positions
        ghosts_position = np.array([0, 0 , 0 , 0], dtype=np.bool_)
        ghosts_rewards = {RIGHT : np.array([0 , 0]) , DOWN : np.array([0 , 0]) , UP : np.array([0 , 0]) , LEFT : np.array([0 , 0])}
        pacman_x_tile = self.pacman.tile[0]
        pacman_y_tile = self.pacman.tile[1]

        #check right direction
        if walls_position[0] == 0:  # the pacman can move to the right
            ghosts_rewards [RIGHT][1] = 101
            for rc in range (pacman_x_tile + 1 , min(pacman_x_tile + 9 , GAME_COLS)):
                ghosts_rewards[RIGHT][0] += self.maze_map[pacman_y_tile][rc][0]  #update the right rewards

                if (self.maze_map[pacman_y_tile][rc][1] != 0):  #detected a ghost on the right
                    ghosts_position[0] = 1  ### update the ghosts_position obs

                    if self.maze_map[pacman_y_tile][rc][1] == 2:   ## the pacman is moving to me to the left
                        if rc - pacman_x_tile < ghosts_rewards[RIGHT][1]:
                            ghosts_rewards[RIGHT][1] =  rc - pacman_x_tile
                    elif 100 < ghosts_rewards[RIGHT][1] :
                            ghosts_rewards[RIGHT][1] = 100

                for up in range (1,min(3 , pacman_y_tile)):
                    ghosts_rewards[RIGHT][0] += self.maze_map[pacman_y_tile - up][rc][0]  #update the right rewards

                    if self.maze_map[pacman_y_tile - up][rc][1] != -0:
                        if self.maze_map[pacman_y_tile - up][rc][1] == -2:    ## the ghost is moving to the right
                            if rc - pacman_x_tile + up < ghosts_rewards[RIGHT][1] :
                                ghosts_rewards[RIGHT][1] = rc - pacman_x_tile + up

                        elif 100 < ghosts_rewards[RIGHT][1]:
                            ghosts_rewards[RIGHT][1] = 100

                for down in range (1,min(3 , GAME_ROWS - pacman_y_tile)):
                    ghosts_rewards[RIGHT][0] += self.maze_map[pacman_y_tile + down][rc][0] #update the right rewards

                    if self.maze_map[pacman_y_tile+down][rc][1] !=0:
                        if self.maze_map[pacman_y_tile+down][rc][1] == 1:   #the ghost is moving up
                            if rc - pacman_x_tile + down < ghosts_rewards[RIGHT][1] :
                                ghosts_rewards[RIGHT][1] = rc - pacman_x_tile + down
                        elif 100 < ghosts_rewards[RIGHT][1]:
                            ghosts_rewards[RIGHT][1] = 100

        #check down
        if walls_position[1] == 0:  # the pacman can move DOWN
            ghosts_rewards [DOWN][1] = 101
            for dc in range (pacman_y_tile + 1 , min(pacman_y_tile + 9 , GAME_ROWS)):
                ghosts_rewards[DOWN][0] += self.maze_map[dc][pacman_x_tile][0]  #update the Down rewards    

                if (self.maze_map[dc][pacman_x_tile][1] != 0):  #detected a ghost DOWN
                    ghosts_position[1] = 1   ### update the ghosts_position obs

                    if (self.maze_map[dc][pacman_x_tile][1] == 1) :    ## the ghost is moving up
                        if dc - pacman_y_tile < ghosts_rewards[DOWN][1] :
                            ghosts_rewards[DOWN][1] =  dc - pacman_y_tile

                    elif 100 < ghosts_rewards[DOWN][1]:
                        ghosts_rewards[DOWN][1] = 100

                for right in range (1,min(3 , GAME_COLS - pacman_x_tile)):
                    ghosts_rewards[DOWN][0] += self.maze_map[dc][pacman_x_tile+right][0] #update the Down rewards

                    if (self.maze_map[dc][pacman_x_tile+right][1] != 0):
                        if (self.maze_map[dc][pacman_x_tile+right][1] == 2):     ## ghost is moving to the left
                            if dc - pacman_y_tile + right < ghosts_rewards[DOWN][1] :
                                ghosts_rewards[DOWN][1] = dc - pacman_y_tile + right
                        elif 100 < ghosts_rewards[DOWN][1]:
                            ghosts_rewards[DOWN][1] = 100

                for left in range (1,min(3 , pacman_x_tile)):
                    ghosts_rewards[DOWN][0] += self.maze_map[dc][pacman_x_tile - left][0] #update the Down rewards

                    if(self.maze_map[dc][pacman_x_tile - left][1] != 0):
                        if (self.maze_map[dc][pacman_x_tile - left][1] == -2):     ##ghost is moving to the right
                            if dc - pacman_y_tile + left < ghosts_rewards[DOWN][1] :
                                ghosts_rewards[DOWN][1] = dc - pacman_y_tile + left
                        elif 100 < ghosts_rewards[DOWN][1]:
                            ghosts_rewards[DOWN][1] = 100

        #check up
        if walls_position[2] == 0:  # the pacman can move up
            ghosts_rewards [UP][1] = 101
            for uc in range (max(pacman_y_tile - 8 , 0) , pacman_y_tile):
                ghosts_rewards[UP][0] += self.maze_map[uc][pacman_x_tile][0]  #update the up rewards

                if (self.maze_map[uc][pacman_x_tile][1] != 0):  #detected a ghost up
                    ghosts_position[2] = 1  ### update the ghosts_position obs

                    if self.maze_map[uc][pacman_x_tile][1] == -1:      ## ghost is moving down
                        if pacman_y_tile - uc < ghosts_rewards[UP][1] :
                            ghosts_rewards[UP][1] =  pacman_y_tile - uc
                    elif 100 < ghosts_rewards[UP][1]:
                        ghosts_rewards[UP][1] = 100

                for right in range (1,min(3 , GAME_COLS - pacman_x_tile)):
                    ghosts_rewards[UP][0] += self.maze_map[uc][pacman_x_tile+right][0]  #update the up rewards

                    if (self.maze_map[uc][pacman_x_tile+right][1] != 0):
                        if (self.maze_map[uc][pacman_x_tile+right][1] == 2):    #ghost is moving to the left
                            if pacman_y_tile - uc + right < ghosts_rewards[UP][1] :
                                ghosts_rewards[UP][1] = pacman_y_tile - uc + right
                        elif 100 < ghosts_rewards[UP][1]:
                            ghosts_rewards[UP][1] = 100

                for left in range (1,min(3 , pacman_x_tile)):
                    ghosts_rewards[UP][0] += self.maze_map[uc][pacman_x_tile-left][0]  #update the up rewards

                    if(self.maze_map[uc][pacman_x_tile - left][1] != 0):
                        if (self.maze_map[uc][pacman_x_tile - left][1] == -2):     #ghost is moving to the right
                            if pacman_y_tile - uc + left < ghosts_rewards[UP][1] :
                                ghosts_rewards[UP][1] = pacman_y_tile - uc + left
                        elif 100 < ghosts_rewards[UP][1] :
                            ghosts_rewards[UP][1] = 100

        #check left direction
        if walls_position[3] == 0:  # the pacman can move to the left
            ghosts_rewards [LEFT][1] = 101
            for lc in range ( max(pacman_x_tile - 8 , 0) , pacman_x_tile):
                ghosts_rewards[LEFT][0] += self.maze_map[pacman_y_tile][lc][0]  #update the left rewards

                if (self.maze_map[pacman_y_tile][lc][1] != 0):  #detected a ghost on the left
                    ghosts_position[3] = 1  ### update the ghosts_position obs

                    if (self.maze_map[pacman_y_tile][lc][1] == -2):    #ghost is moving right
                        if pacman_x_tile - lc < ghosts_rewards[LEFT][1]:
                            ghosts_rewards[LEFT][1] =  pacman_x_tile - lc

                    elif 100 < ghosts_rewards[LEFT][1]:
                        ghosts_rewards[LEFT][1] = 100

                for up in range (1,min(3 , pacman_y_tile)):
                    ghosts_rewards[LEFT][0] += self.maze_map[pacman_y_tile - up][lc][0]  #update the left rewards

                    if self.maze_map[pacman_y_tile - up][lc][1] != 0:
                        if (self.maze_map[pacman_y_tile - up][lc][1] == -1):   ##ghost is moving down
                            if pacman_x_tile -lc + up < ghosts_rewards[LEFT][1]:
                                ghosts_rewards[LEFT][1] = pacman_x_tile - lc + up
                        elif 100 < ghosts_rewards[LEFT][1]:
                            ghosts_rewards[LEFT][1] = 100

                for down in range (1,min(3 , GAME_ROWS - pacman_y_tile)):
                    ghosts_rewards[LEFT][0] += self.maze_map[pacman_y_tile + down][lc][0] #update the left rewards

                    if self.maze_map[pacman_y_tile+down][lc][1] !=0:
                        if self.maze_map[pacman_y_tile+down][lc][1] == 1:   ## ghost is moving up
                            if pacman_x_tile - lc + down < ghosts_rewards[LEFT][1] :
                                ghosts_rewards[LEFT][1] = pacman_x_tile - lc + down
                        elif 100 < ghosts_rewards[LEFT][1]:
                            ghosts_rewards[LEFT][1] = 100

        self.observation = (walls_position , ghosts_position , ghosts_rewards)
        del self.maze_map

    def updateScore(self, points):
        self.RLreward = points
        self.score += points
        self.textgroup.updateScore(self.score)

    def checkEvents(self):
        for event in pygame.event.get():
            if event.type == QUIT:
                exit()
            elif event.type == KEYDOWN:
                if event.key == K_SPACE:
                    if self.pacman.alive:
                        self.pause.setPause(playerPaused=True)
                        if not self.pause.paused:
                            self.textgroup.hideText()
                            self.showEntities()
                        else:
                            self.textgroup.showText(PAUSETXT)
                            self.hideEntities()

    def checkGhostEvents(self):
        for ghost in self.ghosts:
            #### update the maze_map with ghosts penality or reward
            if ghost.mode.current == CHASE or ghost.mode.current == SCATTER:
                self.maze_map[ghost.tile[1]][ghost.tile[0]][1] = ghost.direction
                

            elif ghost.mode.current == FREIGHT :
                self.maze_map[ghost.tile[1]][ghost.tile[0]][0] = ghost.points
            ####

            if self.pacman.collideGhost(ghost):
                if ghost.mode.current is FREIGHT:
                    self.pacman.visible = False
                    ghost.visible = False
                    self.updateScore(ghost.points)
                    self.textgroup.addText(
                        str(ghost.points),
                        WHITE,
                        ghost.position.x,
                        ghost.position.y,
                        8,
                        time=1,
                    )
                    self.ghosts.updatePoints()
                    self.pause.setPause(pauseTime=1, func=self.showEntities)
                    ghost.startSpawn()
                    self.nodes.allowHomeAccess(ghost)
                elif ghost.mode.current is not SPAWN:
                    if self.pacman.alive:
                        self.lives -= 1
                        self.updateScore(GHOST_PENALITY)
                        self.lifesprites.removeImage()
                        self.pacman.die()
                        self.ghosts.hide()
                        if self.lives <= 0:
                            self.textgroup.showText(GAMEOVERTXT)
                            self.gameOver = True
                            self.pause.setPause(pauseTime=3, func=self.restartGame)
                        else:
                            self.pause.setPause(pauseTime=3, func=self.resetLevel)

    def showEntities(self):
        self.pacman.visible = True
        self.ghosts.show()

    def hideEntities(self):
        self.pacman.visible = False
        self.ghosts.hide()

    def checkPelletEvents(self):
        pellet = self.pacman.eatPellets(self.pellets.pelletList)
        if pellet:
            ###update the pellet hen eaten and delete the pellet rewards from it 
            self.set_maze_map[pellet.tile[1]][pellet.tile[0]][0] -= pellet.points
            ###

            self.pellets.numEaten += 1
            self.updateScore(pellet.points)
            if self.pellets.numEaten == 30:
                self.ghosts.inky.startNode.allowAccess(RIGHT, self.ghosts.inky)
            if self.pellets.numEaten == 70:
                self.ghosts.clyde.startNode.allowAccess(LEFT, self.ghosts.clyde)
            self.pellets.pelletList.remove(pellet)
            if pellet.name is POWERPELLET:
                self.ghosts.startFreight()

        if self.pellets.isEmpty():
            self.flashBG = True
            self.hideEntities()
            self.pause.setPause(pauseTime=3, func=self.nextLevel)

    def checkFruitEvents(self):
        if self.pellets.numEaten == 50 or self.pellets.numEaten == 140:
            if self.fruit is None:
                self.fruit = Fruit(self.nodes.getNodeFromTiles(9, 20), self.level)

                #### update the maze map with the fruit reward 
                self.maze_map[self.fruit.tile[1]][self.fruit.tile[0]][0] = self.fruit.points
                ###

        if self.fruit is not None:
            #### update the maze map with the fruit reward 
            self.maze_map[self.fruit.tile[1]][self.fruit.tile[0]][0] = self.fruit.points
            ###
            
            if self.pacman.collideCheck(self.fruit):
                self.updateScore(self.fruit.points)
                self.textgroup.addText(
                    str(self.fruit.points),
                    WHITE,
                    self.fruit.position.x,
                    self.fruit.position.y,
                    8,
                    time=1,
                )
                fruitcaptured = False
                for fruit in self.fruitcaptured:
                    if fruit.get_offset() == self.fruit.image.get_offset():
                        fruitcaptured = True
                        break
                if not fruitcaptured:
                    self.fruitcaptured.append(self.fruit.image)
                self.fruit = None
            elif self.fruit.destroy:
                self.fruit = None

    def render(self):
        self.screen.blit(self.background, (0, 0))
        self.pellets.render(self.screen)
        if self.fruit is not None:
            self.fruit.render(self.screen)
        self.pacman.render(self.screen)
        self.ghosts.render(self.screen)
        self.textgroup.render(self.screen)
        for i in range(len(self.lifesprites.images)):
            x = self.lifesprites.images[i].get_width() * i
            y = SCREENHEIGHT - self.lifesprites.images[i].get_height()
            self.screen.blit(self.lifesprites.images[i], (x, y))
        for i in range(len(self.fruitcaptured)):
            x = SCREENWIDTH - self.fruitcaptured[i].get_width() * (i + 1)
            y = SCREENHEIGHT - self.fruitcaptured[i].get_height()
            self.screen.blit(self.fruitcaptured[i], (x, y))
        pygame.display.update()
9

if __name__ == "__main__":
    game = GameController(rlTraining=True)
    while True:
        game.update(render=True)

        #print(game.maze_map)
