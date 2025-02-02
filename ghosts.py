import pygame
from pygame.locals import *
from vector import Vector2
from constants import *
from entity import Entity
from modes import ModeController
from sprites import GhostSprites


class Ghost(Entity):
    def __init__(self, node, pacman=None, blinky=None):
        Entity.__init__(self, node)
        self.name = GHOST
        self.points = GHOST_REWARD
        self.ghost_penality = GHOST_PENALITY
        self.goal = Vector2()
        self.directionMethod = self.goalDirection
        self.pacman = pacman
        self.mode = ModeController(self)
        self.blinky = blinky
        self.homeNode = node
        self.can_eat = True
        self.can_be_eaten = True
        

    def update(self, dt , agent_direction = None):
        if agent_direction == None:
            self.sprites.update(dt)
            self.mode.update(dt)
            if self.mode.current is SCATTER:
                self.scatter()
            elif self.mode.current is CHASE:
                self.chase()
            Entity.update(self, dt)

        else:
            self.sprites.update(dt)
            direction = agent_direction
            self.initial_tile = self.tile
            self.initial_node = self.node
            self.target = self.getNewTarget(direction)
            self.node = self.target
            if self.node.neighbors[PORTAL] is not None:
                self.node = self.node.neighbors[PORTAL]

            if self.initial_node is not self.node:
                self.direction = direction
            else:
                self.target = self.getNewTarget(self.direction)
                self.node = self.target

            if self.initial_node is self.node:
                self.direction = STOP

            self.setPosition()

    def startFreight(self):
        self.mode.setFreightMode()
        if self.mode.current is FREIGHT:
            self.setSpeed(50)
            self.directionMethod = self.randomDirection

    def normalMode(self):
        self.setSpeed(100)
        self.directionMethod = self.goalDirection
        self.homeNode.denyAccess(DOWN, self)

    def scatter(self):
        self.goal = Vector2()

    def chase(self):
        self.goal = self.pacman.position

    def spawn(self):
        self.goal = self.spawnNode.position

    def setSpawnNode(self, node):
        self.spawnNode = node

    def startSpawn(self):
        self.mode.setSpawnMode()
        if self.mode.current is SPAWN:
            self.setSpeed(150)
            self.directionMethod = self.goalDirection
            self.spawn()


class Blinky(Ghost):
    def __init__(self, node, pacman=None, blinky=None):
        Ghost.__init__(self, node, pacman, blinky)
        self.name = BLINKY
        self.color = RED
        self.sprites = GhostSprites(self)


class Pinky(Ghost):
    def __init__(self, node, pacman=None, blinky=None):
        Ghost.__init__(self, node, pacman, blinky)
        self.name = PINKY
        self.color = PINK
        self.sprites = GhostSprites(self)

    def scatter(self):
        self.goal = Vector2(TILEWIDTH * NCOLS, 0)

    def chase(self):
        self.goal = (
            self.pacman.position
            + self.pacman.directions[self.pacman.direction] * TILEWIDTH * 4
        )


class Inky(Ghost):
    def __init__(self, node, pacman=None, blinky=None):
        Ghost.__init__(self, node, pacman, blinky)
        self.name = INKY
        self.color = TEAL
        self.sprites = GhostSprites(self)

    def scatter(self):
        self.goal = Vector2(TILEWIDTH * NCOLS, TILEHEIGHT * NROWS)

    def chase(self):
        vec1 = (
            self.pacman.position
            + self.pacman.directions[self.pacman.direction] * 2 * TILEWIDTH
        )
        vec2 = (vec1 - self.blinky.position) * 2
        self.goal = self.blinky.position + vec2


class Clyde(Ghost):
    def __init__(self, node, pacman=None, blinky=None):
        Ghost.__init__(self, node, pacman, blinky)
        self.name = CLYDE
        self.color = ORANGE
        self.sprites = GhostSprites(self)

    def scatter(self):
        self.goal = Vector2(0, TILEHEIGHT * NROWS)

    def chase(self):
        d = self.pacman.position - self.position
        ds = d.magnitudeSquared()
        if ds <= (TILEWIDTH * 8) ** 2:
            self.scatter()
        else:
            self.goal = (
                self.pacman.position
                + self.pacman.directions[self.pacman.direction] * TILEWIDTH * 4
            )

    def reset(self):
        Entity.reset(self)
        self.points = GHOST_REWARD
        self.directionMethod = self.goalDirection


class GhostGroup(object):
    def __init__(self, node, pacman , move_mode = DISCRETE_STEPS_MODE , mode = NORMAL_MODE):
        self.blinky = Blinky(node, pacman)
        self.pinky = Pinky(node, pacman)
        self.inky = Inky(node, pacman, self.blinky)
        self.clyde = Clyde(node, pacman)

        if mode == NORMAL_MODE:
            self.ghosts = [self.blinky, self.pinky, self.inky, self.clyde]
        elif mode == SCARY_1_MODE:
            self.ghosts = [self.blinky]
        elif mode == SCARY_2_MODE: 
            self.ghosts = [self.blinky, self.pinky]
        elif mode == SAFE_MODE:
            self.ghosts = []
            
        self.move_mode = move_mode
        self.set_move_mode()

    def __iter__(self):
        return iter(self.ghosts)

    def __getitem__(self, index):
        return self.ghosts[index]

    def set_move_mode(self):
        for ghost in self.ghosts:
            ghost.move_mode = self.move_mode

    def update(self, dt , agents_directions):
        for i , ghost in enumerate(self):
            ghost.update(dt , agents_directions[i])

    def startFreight(self):
        for ghost in self:
            ghost.startFreight()
        self.resetPoints()

    def setSpawnNode(self, node):
        for ghost in self:
            ghost.setSpawnNode(node)

    def updatePoints(self):
        for ghost in self:
            ghost.points += GHOST_UPDATE_REWARD

    def update_penality_points(self):
        for ghost in self:
            ghost.ghost_penality -= GHOST_UPDATE_PENALITY
    # def updatePoints(self):
    #     for ghost in self:
    #         ghost.points *= 2

    def resetPoints(self):
        for ghost in self:
            ghost.points = GHOST_REWARD

    def reset(self):
        for ghost in self:
            ghost.reset()

    def hide(self):
        for ghost in self:
            ghost.visible = False

    def show(self):
        for ghost in self:
            ghost.visible = True

    def render(self, screen):
        for ghost in self:
            ghost.render(screen)