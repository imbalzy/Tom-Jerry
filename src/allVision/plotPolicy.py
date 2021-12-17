
import numpy as np
import matplotlib.pyplot as plt
import gym
from map import Map
from robot import Robot
from qlearning import TurtleBotTag
import datetime
import os
import random


def main():

    # 1. Load Environment and Q-table structure
    env = TurtleBotTag(p_num=1)

    parent_dir = os.getcwd()
    directory1 = "../../results/"
    dir_name = os.path.join(parent_dir, directory1)
    folder = '02_Dec_2021_02_21_51'
    folder += '/model_allVision'
    p_file = dir_name + folder + "/p_model_epi61000"
    e_file = dir_name + folder + "/e_model_epi61000"
    env.load(p_file, e_file)

    env.reset()
    env.plot_p_policy()

if __name__ == '__main__':
    main()
