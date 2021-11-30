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
import random
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from keras import backend as K
import tensorflow as tf

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
THETALIST = [0, np.pi / 2, np.pi, 3 / 2 * np.pi]


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
        self.p_num = p_num
        self.current_grid = self.map.grid

        # Get map bounds
        self.height = self.map.grid_sz[0]
        self.width = self.map.grid_sz[1]
        self.theta = THETA
        self.observations = N_DISCRETE_Z

        # Establish Discrete Actions:
        self.action_space = spaces.Discrete(N_DISCRETE_ACTIONS)
        self.q_action_dim = N_DISCRETE_ACTIONS ** p_num
        self.e_action_dim = N_DISCRETE_ACTIONS

        # self.model
        # (x1, y1, theta1, ..., xn, yn, theta_n, xe, ye, theta_e)
        self.state_size = 3 * (p_num + 1)
        self.hidden_size = 1024
        self.learning_rate = 0.001
        self.q_model = self._build_model(input_size=self.state_size,
                                         hidden_size=self.hidden_size,
                                         output_size=self.q_action_dim,
                                         learning_rate=self.learning_rate,
                                         loss_func=self._huber_loss)
        self.e_model = self._build_model(input_size=self.state_size,
                                         hidden_size=self.hidden_size,
                                         output_size=self.e_action_dim,
                                         learning_rate=self.learning_rate,
                                         loss_func=self._huber_loss)
        self.q_model_target = self._build_model(input_size=self.state_size,
                                                hidden_size=self.hidden_size,
                                                output_size=self.q_action_dim,
                                                learning_rate=self.learning_rate,
                                                loss_func=self._huber_loss)
        self.e_model_target = self._build_model(input_size=self.state_size,
                                                hidden_size=self.hidden_size,
                                                output_size=self.e_action_dim,
                                                learning_rate=self.learning_rate,
                                                loss_func=self._huber_loss)
        self.q_memory = deque(maxlen=2000)
        self.e_memory = deque(maxlen=2000)
        self.gamma = .95
        self.epsilon = 1.0
        self.epsilon_decay = 0.9999
        self.epsilon_min = 0.01

        # Reset Initial Conditions
        self.reset()

    def _huber_loss(self, y_true, y_pred, clip_delta=1.0):
        error = y_true - y_pred
        cond = K.abs(error) <= clip_delta

        squared_loss = 0.5 * K.square(error)
        quadratic_loss = 0.5 * K.square(clip_delta) + clip_delta * (K.abs(error) - clip_delta)

        return K.mean(tf.where(cond, squared_loss, quadratic_loss))

    def _build_model(self, input_size, hidden_size, output_size, learning_rate, loss_func):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(hidden_size, input_dim=input_size, activation='relu'))
        model.add(Dense(hidden_size, activation='relu'))
        model.add(Dense(output_size, activation='linear'))
        model.compile(loss=loss_func,
                      optimizer=Adam(learning_rate=learning_rate))
        return model

    def update_target_model(self):
        # copy weights from model to target_model
        self.q_model_target.set_weights(self.q_model.get_weights())
        self.e_model_target.set_weights(self.e_model.get_weights())

    def memorize(self, state, action, reward, next_state, done, role):
        memory = self.q_memory if role == 'p' else self.e_memory
        memory.append([np.array(state), action, reward, np.array(next_state), done])

    def load(self, q_name, e_name):
        self.q_model.load_weights(q_name)
        self.e_model.load_weights(e_name)

    def save(self, q_name, e_name):
        self.q_model.save_weights(q_name)
        self.e_model.save_weights(e_name)

    def act(self, q_state, e_state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.q_action_dim), random.randrange(self.e_action_dim)
        q_action_values = self.q_model.predict(np.expand_dims(np.asarray(q_state), axis=0))
        e_action_values = self.e_model.predict(np.expand_dims(np.asarray(e_state), axis=0))
        return np.argmax(q_action_values[0]), np.argmax(e_action_values[0])

    def replay(self, batch_size, role):
        if role == 'p':
            memory = self.q_memory
            model = self.q_model
            target_model = self.q_model_target
        else:
            memory = self.e_memory
            model = self.e_model
            target_model = self.e_model_target

        states = []
        next_states = []
        for m in memory:
            states.append(m[0])
            next_states.append(m[3])
        states = np.array(states)
        next_states = np.array(next_states)
        targets = model.predict(states)
        t = target_model.predict(next_states)

        for i, (state, action, reward, next_state, done) in enumerate(memory):
            # state = np.reshape(state, [-1, self.state_size])
            # next_state = np.reshape(next_state, [-1, self.state_size])
            # target = model.predict([state])
            # if done:
            #     target[0][action] = reward
            # else:
            #     t = target_model.predict([next_state])[0]
            #     target[0][action] = reward + self.gamma * np.amax(t)
            if done:
                targets[i][action] = reward
            else:
                targets[i][action] = reward + self.gamma * np.amax(t[i])

            # if (role == 'p' and sum(state[0][-3:]) > 0) or (role == 'e' and sum(state[0][:-3])):
            #     print(role, state)

        model.fit(states, targets, epochs=1, verbose=0)
        memory.clear()

    def decodePAction(self, a):
        actions = []
        while len(actions) < self.p_num:
            actions.append(a % N_DISCRETE_ACTIONS)
            a = a // N_DISCRETE_ACTIONS
        return actions

    def step(self, p_action, e_action):
        # Execute one time step within the environment
        # pursuer and evader sense and then move
        # update the map with their pose

        p_actions = self.decodePAction(p_action)

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

        p_observation = self.map.pursuerScanner()  # 0 is get; 1 is not
        e_observation = self.map.evaderScanner()  # [p1, p2, p3]

        if self.p_observation > 0:
            p_state = tuple(p_poses + list(e_pose))
        else:
            p_state = tuple(p_poses + [0, 0, 0])
        e_state = tuple(self.e_observation + list(e_pose))

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
        poses = self.generateRandomPos(self.map.p_num + 1)
        p_poses = []
        for i, p in enumerate(self.map.r_p):
            p.pose = poses[i]
            p_poses += list(p.pose)
        self.map.r_e.pose = tuple(poses[-1])  # numpy random (x, y, theta)
        self.p_observation = self.map.pursuerScanner()
        self.e_observation = self.map.evaderScanner()

        if self.p_observation > 0:
            p_state = tuple(p_poses + list(self.map.r_e.pose))
        else:
            p_state = tuple(p_poses + [0, 0, 0])
        e_state = tuple(self.e_observation + list(self.map.r_e.pose))
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

            if not self.map.checkForObstacle(x, y) and (x, y) not in seen:
                theta = np.random.choice(thetaRange)
                ret.append([x, y, theta])
                seen.add((x, y))
        return ret

    def render(self):
        if self.PRINT_CONSOLE and self.step_num == 0 and self.epis % 10 == 0:
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

            e_theta1 = 180 / np.pi * (self.map.r_e.THETALIST[self.map.r_e.pose[2]] - self.map.r_e.fov / 2)
            e_theta2 = 180 / np.pi * (self.map.r_e.THETALIST[self.map.r_e.pose[2]] + self.map.r_e.fov / 2)
            e_wedge = Wedge((self.map.r_e.pose[0], self.map.r_e.pose[1]), self.map.r_e.VIEW_DIST, e_theta1, e_theta2,
                            facecolor='r')
            patches.append(e_wedge)

            p = PatchCollection(patches, alpha=0.8)
            ax = plt.gca()
            ax.add_collection(p)

            plt.legend(loc='lower left')
            plt.title('Episode: ' + str(self.epis) + '    Step: ' + str(self.step_num))
            # Show the graph without blocking the rest of the program

            plt.draw()
            if self.SAVE_PLOTS:
                fname = self.dir_name_plots + '/epis' + str(self.epis).zfill(5) + '_step' + str(self.step_num).zfill(5)
                plt.savefig(fname)
            plt.pause(0.005)
