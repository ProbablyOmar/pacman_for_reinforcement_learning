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

    def restartGame(self):
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
        self.pause.paused = not self.rlTraining
        self.pacman.reset()
        self.ghosts.reset()
        self.fruit = None
        if not (self.rlTraining):
            self.textgroup.showText(READYTXT)

    def nextLevel(self):
        self.showEntities()
        if self.level > NUMBEROFLEVELS:
            self.gameOver = True
        self.pause.pasued = True
        self.restartGame()
        self.level += 1
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

    def update(self, agent_direction=None, render=True, clocktick=60):
        dt = self.clock.tick(clocktick) / 1000.0
        self.textgroup.update(dt)
        self.pellets.update(dt)
        if not self.pause.paused:
            self.ghosts.update(dt)
            if self.fruit is not None:
                self.fruit.update(dt)
            self.checkPelletEvents()
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

        afterPauseMethod = self.pause.update(dt)
        if afterPauseMethod is not None:
            afterPauseMethod()
        self.checkEvents()
        if render:
            self.render()

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
                        self.updateScore(-50)
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
        if self.fruit is not None:
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
    game.startGame()
    while True:
        game.update(render=True)
