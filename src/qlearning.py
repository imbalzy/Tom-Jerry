# Source for Q-learning implementation: https://towardsdatascience.com/reinforcement-learning-with-openai-d445c2c687d2
# Source for cutom Environment Creation: https://towardsdatascience.com/creating-a-custom-openai-gym-environment-for-stock-trading-be532be3910e

import gym
import numpy as np
from map import Map
from robot import Robot
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Wedge, Polygon
from matplotlib.collections import PatchCollection
from gym import spaces, error, utils
from gym.utils import seeding
import time
import os

# linear vel = [0, 1]
# angular vel = [-1, 0, 1]
ACTION_LIST_NAMES = ["stop", "forward", "turn_left", "turn_right", "turn_and_move_left", "turn_and_move_right"]
ACTION_LIST = [(0, 0), (1, 0), (0, -1), (0, 1), (1, -1), (1, 1)]
N_DISCRETE_ACTIONS = len(ACTION_LIST)

Z_LIST = [0, 1]
N_DISCRETE_Z = len(Z_LIST)

HEIGHT = 100
WIDTH = 100
THETA = 4
THETALIST = [0, np.pi/2, np.pi, 3/2*np.pi]


class TurtleBotTag(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}
    epis = 0
    step_num = 0
    RENDER_FREQ = 1
    PRINT_CONSOLE = True
    RENDER_PLOTS = True
    SAVE_PLOTS = True

    def __init__(self, p_num):
        super(TurtleBotTag, self).__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        self.fig = plt.figure()
        self.ax = self.fig.axes

        parent_dir = os.getcwd()
        curr_time = time.gmtime()
        directory1 = time.strftime("../results/%d_%b_%Y_%H_%M_%S", curr_time)
        self.dir_name = os.path.join(parent_dir, directory1)
        os.mkdir(self.dir_name)

        if self.SAVE_PLOTS:
            directory = time.strftime("../results/%d_%b_%Y_%H_%M_%S/figures", curr_time)
            self.dir_name_plots = os.path.join(parent_dir, directory)
            os.mkdir(self.dir_name_plots)

        # Obtain current map for training
        self.map = Map(p_num=p_num)
        self.current_grid = self.map.grid

        # Get map bounds
        self.height = self.map.grid_sz[0]
        self.width = self.map.grid_sz[1]
        self.theta = THETA
        self.observations = N_DISCRETE_Z

        # Establish Discrete Actions:
        self.action_space = spaces.Discrete(N_DISCRETE_ACTIONS)

        # Establish Observation Space: ADD ROBOT POSE
        self.observation_space = spaces.Box(low=0, high=1, shape=
        (self.height, self.width, self.theta, self.observations), dtype=np.uint8)
        self.Q_dim = (self.height, self.width, self.theta, self.height, self.width, self.theta, self.height, self.width, self.theta, self.height, self.width, self.theta, self.action_space.n)

        # self.observation_space = spaces.Box(low=0, high=width, shape=
        #                (HEIGHT, WIDTH), dtype=np.uint8)

        # Reset Initial Conditions
        self.reset()

    def step(self, p_actions, e_action):
        # Execute one time step within the environment
        # pursuer and evader sense and then move
        # update the map with their pose

        # MOVE ROBOTS
        p_vels = []
        for p_action in p_actions:
            p_vels.append(ACTION_LIST[p_action])
        e_vel = ACTION_LIST[e_action]

        p_poses = []
        for i, p_vel in enumerate(p_vels):
            p = self.map.r_p[i]
            p_poses += list(p.move([p_vel[0], p_vel[1]], self.map))
        e_pose = self.map.r_e.move([e_vel[0], e_vel[1]], self.map)

        p_observation = self.map.pursuerScanner() # 0 is get; 1 is not
        e_observation = self.map.evaderScanner() # [p1, p2, p3]

        if self.p_observation > 0:
            p_state = tuple(p_poses+list(e_pose))
        else:
            p_state = tuple(p_poses + [0, 0, 0])
        e_state = tuple(self.e_observation+list(e_pose))

        done = self.map.haveCollided()

        p_reward = 0
        e_reward = 0

        # reward observation of other robot
        if p_observation > 0:
            p_reward += 2
        if e_observation == 1:
            e_reward += -2

        if done:
            p_reward += 100
            e_reward += -100
        else:
            p_reward += -1
            e_reward += 1

        return p_state, e_state, p_reward, e_reward, done

    def reset(self):
        # Reset the state of the environment to an initial state
        # reset to random robot starting positions so we can make sure they
        # all converge to the same solution
        poses = self.generateRandomPos(self.map.p_num+1)
        p_poses = []
        for i, p in enumerate(self.map.r_p):
            p.pose = poses[i]
            p_poses += list(p.pose)
        self.map.r_e.pose = tuple(poses[-1])  # numpy random (x, y, theta)
        self.p_observation = self.map.pursuerScanner()
        self.e_observation = self.map.evaderScanner()

        if self.p_observation > 0:
            p_state = tuple(p_poses+list(self.map.r_e.pose))
        else:
            p_state = tuple(p_poses + [0, 0, 0])
        e_state = tuple(self.e_observation+list(self.map.r_e.pose))
        return p_state, e_state

    # define states as state =  (y, x, theta)

    def generateRandomPos(self, num):
        heightRange = range(self.width)
        widthRange = range(self.width)
        thetaRange = range(self.theta)

        seen = set()
        ret = []
        while len(ret) != num:
            # randomize position
            x = np.random.choice(widthRange)
            y = np.random.choice(heightRange)

            if not self.map.checkForObstacle(x, y) and (x,y) not in seen:
                theta = np.random.choice(thetaRange)
                ret.append([x, y, theta])
                seen.add((x, y))
        return ret

    def render(self):
        if self.PRINT_CONSOLE and self.step_num == 0 and self.epis%10 == 0:
            print('Episode: ' + str(self.epis))

        if self.RENDER_PLOTS and self.epis % self.RENDER_FREQ == 0:
            # Render the environment to the screen
            plt.cla()
            plt.imshow(self.map.grid, origin='lower')

            patches = []
            for p in self.map.r_p:
                plt.plot(p.pose[0], p.pose[1], 'bo', label='Pursuer')
                p_theta1 = 180 / np.pi * (p.THETALIST[p.pose[2]] - p.fov / 2)
                p_theta2 = 180 / np.pi * (p.THETALIST[p.pose[2]] + p.fov / 2)
                p_wedge = Wedge((p.pose[0], p.pose[1]), p.VIEW_DIST, p_theta1,
                                p_theta2, facecolor='b')
                patches.append(p_wedge)

            plt.plot(self.map.r_e.pose[0], self.map.r_e.pose[1], 'ro', label='Evader')

            e_theta1 = 180/np.pi*(self.map.r_e.THETALIST[self.map.r_e.pose[2]] - self.map.r_e.fov / 2)
            e_theta2 = 180/np.pi*(self.map.r_e.THETALIST[self.map.r_e.pose[2]] + self.map.r_e.fov / 2)
            e_wedge = Wedge((self.map.r_e.pose[0], self.map.r_e.pose[1]), self.map.r_e.VIEW_DIST, e_theta1, e_theta2, facecolor='r')
            patches.append(e_wedge)

            p = PatchCollection(patches, alpha=0.8)
            ax = plt.gca()
            ax.add_collection(p)

            plt.legend(loc='lower left')
            plt.title('Episode: ' + str(self.epis) + '    Step: ' + str(self.step_num))
            # Show the graph without blocking the rest of the program

            plt.draw()
            if self.SAVE_PLOTS:
                fname = self.dir_name_plots + '/epis' + str(self.epis).zfill(5) +'_step' +str(self.step_num).zfill(5)
                plt.savefig(fname)
            plt.pause(0.005)
        else:
            exit(0)
