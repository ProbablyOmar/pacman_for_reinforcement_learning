import pygame
from pygame.locals import *
from vector import Vector2
from constants import *
from entity import Entity
from sprites import PacmanSprites


class Pacman(Entity):
    def __init__(self, node , move_mode = DISCRETE_STEPS_MODE):
        Entity.__init__(self, node)
        self.name = PACMAN
        self.color = YELLOW
        self.direction = LEFT
        self.setBetweenNodes(LEFT)
        self.alive = True
        self.sprites = PacmanSprites(self)
        self.can_eat = True
        self.move_mode = move_mode


    def reset(self):
        Entity.reset(self)
        self.can_eat = True
        self.direction = LEFT
        self.setBetweenNodes(LEFT)
        self.alive = True
        self.image = self.sprites.getStartImage()
        self.sprites.reset()

    def die(self):
        self.alive = False
        self.direction = STOP

    def eatPellets(self, pelletList):
        if self.can_eat:
            for pellet in pelletList:
                if self.tile_collideCheck(pellet):
                    return pellet
        return None

    def collideCheck(self, other):
        d = self.position - other.position
        dsquared = d.magnitudeSquared()
        rsquared = (other.radius + self.collideRadius) ** 2
        if dsquared <= rsquared:
            return True
        return False

    def tile_collideCheck(self, other):
        replace_places = False

        if other.name != PELLET and other.name != POWERPELLET:
            replace_places = self.initial_tile == other.tile and self.tile == other.initial_tile

        if self.tile == other.tile  or replace_places:
            return True
        return False

    def collideGhost(self, ghost):
        return self.tile_collideCheck(ghost)

    def update(self, dt, agent_direction=None):
        if self.move_mode == DISCRETE_STEPS_MODE:
            # self.sprites.update(dt)
            # #self.position += self.directions[self.direction] * self.speed * dt
            # if agent_direction == None:
            #     direction = self.getValidKey()
            # else:
            #     direction = agent_direction
            # #if self.overshotTarget():

            # self.node = self.getNewTarget(direction)
            # if self.node.neighbors[PORTAL] is not None:
            #     self.node = self.node.neighbors[PORTAL]
            
            # if self.target is not self.node:
            #     self.direction = direction
            # else:
            #     self.target = self.getNewTarget(self.direction)

            # if self.target == self.node:
            #     self.direction = STOP
            # self.setPosition()
            #else:
            # if self.oppositeDirection(direction):
            #     self.reverseDirection()
            self.sprites.update(dt)
            #self.position += self.directions[self.direction] * self.speed * dt

            if agent_direction == None:
                direction = self.getValidKey()
            else:
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
            # current_tile = (int((self.node.position.x // TILEWIDTH)), int((self.node.position.y // TILEHEIGHT) - 3)) 
            # target_tile = (int((self.target.position.x // TILEWIDTH)), int((self.target.position.y // TILEHEIGHT) - 3))
            # # target_tile = (int((self.target.position.x // TILEWIDTH)), int((self.target.position.y // TILEHEIGHT) - 3))
            # print(current_tile , target_tile)

        elif self.move_mode == CONT_STEPS_MODE:
            self.sprites.update(dt)
            self.position += self.directions[self.direction] * self.speed * dt
            if agent_direction == None:
                direction = self.getValidKey()
            else:
                direction = agent_direction
            if self.overshotTarget():
                self.node = self.target
                if self.node.neighbors[PORTAL] is not None:
                    self.node = self.node.neighbors[PORTAL]
                self.target = self.getNewTarget(direction)
                if self.target is not self.node:
                    self.direction = direction
                else:
                    self.target = self.getNewTarget(self.direction)

                if self.target == self.node:
                    self.direction = STOP
                self.setPosition()
            else:
                if self.oppositeDirection(direction):
                    self.reverseDirection()


    def getValidKey(self):
        key_pressed = pygame.key.get_pressed()
        if key_pressed[K_UP]:
            return UP
        if key_pressed[K_DOWN]:
            return DOWN
        if key_pressed[K_LEFT]:
            return LEFT
        if key_pressed[K_RIGHT]:
            return RIGHT
        return STOP
