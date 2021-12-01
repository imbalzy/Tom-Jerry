import numpy as np
from robot import Robot
from collections import deque


class Map:

    # Class Attribute
    grid_sz = (8, 8)
    fov = np.pi
    #grid
    #r_p # pursuer robot object
    #r_e # evader robot object

    # Initializer / Instance Attributes
    # Creates grid
    def __init__(self, p_num):
        self.grid = np.zeros(self.grid_sz)
        for i in range(self.grid_sz[0]//4):
            for j in range(self.grid_sz[1]//4):
                self.grid[(i*4):(i*4)+2, (j*4):(j*4)+2] = np.ones((2, 2))

        self.p_num = p_num
        self.r_p = [Robot((0, 0, 0), self.fov) for _ in range(p_num)]
        self.r_e = Robot((0, 0, 0), self.fov)
        print('start build distance')
        self.buildDistance()
        print('finish build distance')

    def buildDistance(self):
        self.dist = {}
        for i in range(self.grid_sz[0]):
            for j in range(self.grid_sz[1]):
                self.bfs(i, j)

    def bfs(self, starti, startj):
        distance = {}
        q = deque([(starti, startj, 0)])
        seen = set()
        seen.add((starti, startj))
        while q:
            curi, curj, curDis = q.popleft()
            distance[(curi, curj)] = curDis
            d = [[-1, 0], [1, 0], [0, -1], [0, 1]]
            for di, dj in d:
                newi, newj, newDis = curi + di, curj + dj, curDis + 1
                if self.validSpace((newi, newj)) and (newi, newj) not in seen:
                    q.append((newi, newj, newDis))
                    seen.add((newi, newj))
        self.dist[(starti, startj)] = distance

    # Checks if input is in an open space in Map
    def validSpace(self, pose):
        x_pos = pose[0]//1
        y_pos = pose[1]//1

        if x_pos < 0 or x_pos >= self.grid_sz[0] or y_pos < 0 or y_pos >= self.grid_sz[1]:
            return False
        elif self.grid[x_pos, y_pos] == 1:
            return False
        else:
            return True

    # def pursuerScanner(self):
    #     # 0 stands for sensing the e; 1 stands for not
    #     # as far as one can sense, the whole can sense.
    #     return sum([p.senseRobot(self.r_e.pose) for p in self.r_p])
    #
    # def evaderScanner(self):
    #     # return the sensed p positions
    #     ret = [0 for _ in range(self.p_num*3)]
    #     sense_num = 0
    #     for p in self.r_p:
    #         if self.r_e.senseRobot(p.pose):
    #             ret[sense_num*3:(sense_num+1)*3] = p.pose
    #             sense_num += 1
    #     return ret

    def haveCollided(self):
        for p in self.r_p:
            dist = self.getDist(p.pose, self.r_e.pose)
            if dist <= 1:
                return True
        return False

    def getDist(self, pose1, pose2):
        return np.sqrt(np.square(pose1[0]-pose2[0]) + np.square(pose1[1]-pose2[1]))

    def checkForObstacle(self, x, y):
        x = int(x)
        y = int(y)

        # check if the desired next state is out of bounds or a wall ( == 1)
        if y > self.grid.shape[0] - 1 or y < 0 or x > self.grid.shape[0] - 1 or x < 0:
            # return initial state if out of bounds
            return True
        elif self.grid[x, y] == 1:
            # return initial state if obstacle found
            return True
        else:
            # return next state if no obstacle found
            return False


    # Does nothing
    # Returns state of surrounding tiles (no robot)
    def getSurroundings(self, pose):
        # 0  open, 1 wall/outofbounds, 2 Robot
        # List (Up, down, left, right)
        x_pos = pose[0]//1
        y_pos = pose[1]//1

        surrounding = []
        #up (i-1, j)
        if x_pos-1 < 0 or self.grid[x_pos-1,y_pos] == 1:
            surrounding[0] = 1
        else:
            surrounding[0] = 0

        #down (i+1, j)
        if x_pos+1 >= self.grid_sz[0] or self.grid[x_pos+1,y_pos] == 1:
            surrounding[1] = 1
        else:
            surrounding[1] = 0

        #left (i, j-1)
        if y_pos-1 < 0 or self.grid[x_pos,y_pos-1] == 1:
            surrounding[2] = 1
        else:
            surrounding[2] = 0

        #right (i, j+1)
        if y_pos+1 >= self.grid_sz[1] or self.grid[x_pos,y_pos+1] == 1:
            surrounding[3] = 1
        else:
            surrounding[3] = 0

        return surrounding
