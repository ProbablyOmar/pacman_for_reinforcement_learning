from constants import *
import random


class MazeBase(object):
    def __init__(self):
        self.portalPairs = {}
        self.homeoffset = (0, 0)
        self.ghostNodeDeny = {UP: (), DOWN: (), LEFT: (), RIGHT: ()}

    def setPortalPairs(self, nodes):
        for pair in list(self.portalPairs.values()):
            nodes.setPortalPair(*pair)

    def connectHomeNodes(self, nodes):
        key = nodes.createHomeNodes(*self.homeoffset)
        nodes.connectHomeNodes(key, self.homenodeconnectLeft, LEFT)
        nodes.connectHomeNodes(key, self.homenodeconnectRight, RIGHT)

    def addOffset(self, x, y):
        return x + self.homeoffset[0], y + self.homeoffset[1]

    def denyGhostsAccess(self, ghosts, nodes):
        nodes.denyAccessList(*(self.addOffset(2, 3) + (LEFT, ghosts)))
        nodes.denyAccessList(*(self.addOffset(2, 3) + (RIGHT, ghosts)))

        for direction in list(self.ghostNodeDeny.keys()):
            for values in self.ghostNodeDeny[direction]:
                nodes.denyAccessList(*(values + (direction, ghosts)))


class Maze1(MazeBase):
    def __init__(self , pac_pos_mode):
        MazeBase.__init__(self)
        self.pac_pos_mode = pac_pos_mode
        self.name = "maze1"
        self.portalPairs = {0: ((0, 17), (27, 17))}
        self.homeoffset = (11.5, 14)
        self.homenodeconnectLeft = (12, 14)     #(12, 14)
        self.homenodeconnectRight = (14, 14)

        if self.pac_pos_mode  == NORMAL_PAC_POS:
            self.pacmanStart = (14, 26)    #(14, 26)
        elif self.pac_pos_mode  == RANDOM_PAC_POS:
            self.pacmanStart = random.choice([(14,26) , (2, 26) , (2, 4) , (26, 4) , ((26, 32))])

        self.fruitStart = (9, 20)
        self.ghostNodeDeny = {
            UP: ((12, 14), (15, 14), (12, 26), (15, 26)),
            LEFT: (self.addOffset(2, 3),),
            RIGHT: (self.addOffset(2, 3),),
        }

class Maze3(MazeBase):
    def __init__(self , pac_pos_mode):
        MazeBase.__init__(self)
        self.pac_pos_mode = pac_pos_mode
        self.name = "maze3"
        self.portalPairs = {0: ((0, 17), (27, 17))}
        self.homeoffset = (11.5, 14)
        self.homenodeconnectLeft = (12, 14)     #(12, 14)
        self.homenodeconnectRight = (14, 14)
        if self.pac_pos_mode  == NORMAL_PAC_POS:
            self.pacmanStart = (14, 26)    #(14, 26)
        elif self.pac_pos_mode  == RANDOM_PAC_POS:
            self.pacmanStart = random.choice([(14,26) , (2, 26) , (2, 4) , (26, 4) , ((26, 32))])
        self.fruitStart = (9, 20)
        self.ghostNodeDeny = {
            UP: ((12, 14), (15, 14), (12, 26), (15, 26)),
            LEFT: (self.addOffset(2, 3),),
            RIGHT: (self.addOffset(2, 3),),
        }

class Maze4(MazeBase):
    def __init__(self , pac_pos_mode):
        MazeBase.__init__(self)
        self.pac_pos_mode = pac_pos_mode
        self.name = "maze4"
        self.portalPairs = {0: ((0, 17), (27, 17))}
        self.homeoffset = (11.5, 14)
        self.homenodeconnectLeft = (12, 14)     #(12, 14)
        self.homenodeconnectRight = (14, 14)
        if self.pac_pos_mode  == NORMAL_PAC_POS:
            self.pacmanStart = (14, 26)    #(14, 26)
        elif self.pac_pos_mode  == RANDOM_PAC_POS:
            self.pacmanStart = random.choice([(14,26) , (2, 26) , (2, 4) , (26, 4) , ((26, 32))])
        self.fruitStart = (9, 20)
        self.ghostNodeDeny = {
            UP: ((12, 14), (15, 14), (12, 26), (15, 26)),
            LEFT: (self.addOffset(2, 3),),
            RIGHT: (self.addOffset(2, 3),),
        }

class Maze5(MazeBase):
    def __init__(self , pac_pos_mode):
        MazeBase.__init__(self)
        self.pac_pos_mode = pac_pos_mode
        self.name = "maze5"
        self.portalPairs = {0: ((0, 17), (27, 17))}
        self.homeoffset = (11.5, 14)
        self.homenodeconnectLeft = (12, 14)     #(12, 14)
        self.homenodeconnectRight = (14, 14)
        if self.pac_pos_mode  == NORMAL_PAC_POS:
            self.pacmanStart = (14, 26)    #(14, 26)
        elif self.pac_pos_mode  == RANDOM_PAC_POS:
            self.pacmanStart = random.choice([(14,26) , (2, 26) , (2, 4) , (26, 4) , ((26, 32))])
        self.fruitStart = (9, 20)
        self.ghostNodeDeny = {
            UP: ((12, 14), (15, 14), (12, 26), (15, 26)),
            LEFT: (self.addOffset(2, 3),),
            RIGHT: (self.addOffset(2, 3),),
        }

class Maze6(MazeBase):
    def __init__(self , pac_pos_mode):
        MazeBase.__init__(self)
        self.pac_pos_mode = pac_pos_mode
        self.name = "maze6"
        self.portalPairs = {0: ((0, 17), (27, 17))}
        self.homeoffset = (11.5, 14)
        self.homenodeconnectLeft = (12, 14)     #(12, 14)
        self.homenodeconnectRight = (14, 14)
        if self.pac_pos_mode  == NORMAL_PAC_POS:
            self.pacmanStart = (14, 26)    #(14, 26)
        elif self.pac_pos_mode  == RANDOM_PAC_POS:
            self.pacmanStart = random.choice([(14,26) , (2, 26) , (2, 4) , (26, 4) , ((26, 32))])
        self.fruitStart = (9, 20)
        self.ghostNodeDeny = {
            UP: ((12, 14), (15, 14), (12, 26), (15, 26)),
            LEFT: (self.addOffset(2, 3),),
            RIGHT: (self.addOffset(2, 3),),
        }

class Maze2(MazeBase):
    def __init__(self):
        MazeBase.__init__(self)
        self.name = "maze2"
        self.portalPairs = {0: ((0, 4), (27, 4)), 1: ((0, 26), (27, 26))}
        self.homeoffset = (11.5, 14)
        self.homenodeconnectLeft = (9, 14)
        self.homenodeconnectRight = (18, 14)
        self.pacmanStart = (16, 26)
        self.fruitStart = (11, 20)
        self.ghostNodeDeny = {
            UP: ((9, 14), (18, 14), (11, 23), (16, 23)),
            LEFT: (self.addOffset(2, 3),),
            RIGHT: (self.addOffset(2, 3),),
        }


class MazeData(object):
    def __init__(self):
        self.obj = None
        self.mazedict = {0: Maze1, 1: Maze2 , 2 : Maze3 , 3: Maze4 , 4:Maze5 , 5:Maze6}

    def loadMaze(self, maze_mode , pac_pos_mode):
        if maze_mode == RAND_MAZE:
            rand_ch = [i for i in range(len(self.mazedict)) if i != 1]
            rand_idx = random.choice(rand_ch)
            self.obj = self.mazedict[rand_idx](pac_pos_mode)
        else:
            self.obj = self.mazedict[maze_mode % (len(self.mazedict))](pac_pos_mode)
