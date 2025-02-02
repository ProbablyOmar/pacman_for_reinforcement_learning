import pygame
from pygame.locals import *
from vector import Vector2
from constants import *
from random import randint


class Entity(object):
    def __init__(self, node , move_mode = DISCRETE_STEPS_MODE):
        self.name = None
        self.directions = {
            UP: Vector2(0, -1),
            DOWN: Vector2(0, 1),
            LEFT: Vector2(-1, 0),
            RIGHT: Vector2(1, 0),
            STOP: Vector2(),
        }
        self.direction = STOP
        self.setSpeed(100)
        self.radius = 10
        self.collideRadius = 5
        self.color = WHITE
        self.visible = True
        self.disablePortal = False
        self.goal = None
        self.directionMethod = self.randomDirection
        self.setStartNode(node)
        self.image = None
        self.move_mode = move_mode

    def setPosition(self):
        self.position = self.node.position.copy()
        self.tile = (int((self.position.x // TILEWIDTH)), int((self.position.y // TILEHEIGHT) - 3))

    def setStartNode(self, node):
        self.node = node
        self.startNode = node
        self.target = node
        self.setPosition()

    def update(self, dt):
        if self.move_mode == CONT_STEPS_MODE:
            self.position += self.directions[self.direction] * self.speed * dt

            if self.overshotTarget():
                self.node = self.target
                directions = self.validDirections()
                direction = self.directionMethod(directions)
                if not self.disablePortal:
                    if self.node.neighbors[PORTAL] is not None:
                        self.node = self.node.neighbors[PORTAL]
                self.target = self.getNewTarget(direction)
                if self.target is not self.node:
                    self.direction = direction
                else:
                    self.target = self.getNewTarget(self.direction)
                self.setPosition()

        elif self.move_mode == DISCRETE_STEPS_MODE:
            #self.position += self.directions[self.direction] * self.speed * dt

            #if self.overshotTarget():
            self.initial_tile = self.tile
            self.node = self.target
            directions = self.validDirections()
            direction = self.directionMethod(directions)
            if not self.disablePortal:
                if self.node.neighbors[PORTAL] is not None:
                    self.node = self.node.neighbors[PORTAL]
            self.target = self.getNewTarget(direction)
            if self.target is not self.node:
                self.direction = direction
            else:
                self.target = self.getNewTarget(self.direction)
            self.setPosition()

    def goalDirection(self, directions):
        distances = []
        for direction in directions:
            vec = self.position + self.directions[direction] * TILEWIDTH - self.goal
            distances.append(vec.magnitudeSquared())
        index = distances.index(min(distances))
        return directions[index]

    def validDirections(self):
        directions = []
        for key in [UP, DOWN, LEFT, RIGHT]:
            if self.validDirection(key):
                if key != self.direction * -1:
                    directions.append(key)
        if len(directions) == 0:
            directions.append(self.direction * -1)
        return directions
    """
    def validAgentDirections(self):
        directions = []
        for key in [UP, LEFT]:
            if self.validDirection(key):
                directions.append(key+1)
        for key in [RIGHT , DOWN]:
            if self.validDirection(key):
                directions.append(key+2)
        return directions
    """
    def randomDirection(self, directions):
        return directions[randint(0, len(directions) - 1)]

    def validDirection(self, direction):
        if direction is not STOP:
            if self.name in self.node.access[direction]:
                if self.node.neighbors[direction] is not None:
                    return True
        return False

    def getNewTarget(self, direction):
        if self.validDirection(direction):
            return self.node.neighbors[direction]
        return self.node

    def overshotTarget(self):
        if self.target is not None:
            vec1 = self.target.position - self.node.position
            vec2 = self.position - self.node.position
            node2Target = vec1.magnitudeSquared()
            node2Self = vec2.magnitudeSquared()
            return node2Self >= node2Target
        return False

    def reverseDirection(self):
        self.direction *= -1
        temp = self.node
        self.node = self.target
        self.target = temp

    def reset(self):
        self.setStartNode(self.startNode)
        self.direction = STOP
        self.speed = 100
        self.visible = True

    def oppositeDirection(self, direction):
        if direction is not STOP:
            if direction == self.direction * -1:
                return True
        return False

    def setBetweenNodes(self, direction): 
        if self.node.neighbors[direction] is not None:
            self.target = self.node.neighbors[direction]
            self.position = self.node.position 
            self.tile = (int((self.position.x // TILEWIDTH)), int((self.position.y // TILEHEIGHT) - 3)) 

    def setSpeed(self, speed):
        self.speed = speed * TILEWIDTH / 16

    def render(self, screen):
        if self.visible:
            if self.image is not None:
                adjust = Vector2(TILEWIDTH, TILEHEIGHT) / 2
                p = self.position - adjust
                screen.blit(self.image, p.asTuple())
            else:
                p = self.position.asInt()
                pygame.draw.circle(screen, self.color, p, self.radius)

    def hit_wall (self , maze_map , direction):
        entity_x_tile , entity_y_tile = self.tile
        if direction == RIGHT and entity_x_tile < GAME_COLS - 1 and maze_map[entity_y_tile][entity_x_tile+1] == WALL_MAZE:
            return True
        elif direction == LEFT and entity_x_tile > 0 and maze_map[entity_y_tile][entity_x_tile-1] == WALL_MAZE:
            return True
        elif direction == UP and entity_y_tile > 0 and maze_map[entity_y_tile - 1][entity_x_tile] == WALL_MAZE:
            return True
        elif direction == DOWN and entity_y_tile < GAME_ROWS -1 and maze_map[entity_y_tile + 1][entity_x_tile] == WALL_MAZE:
            return True
        else:
            #print(direction)
            return False
        
