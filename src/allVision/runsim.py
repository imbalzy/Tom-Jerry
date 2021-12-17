import numpy as np
import matplotlib.pyplot as plt
import gym
from map import Map
from robot import Robot
from qlearning import TurtleBotTag
import datetime
import os
import random

def opsite_action(a):
    if a == 0:
        return 1
    if a == 1:
        return 2
    if a == 2:
        return 1
    if a == 3:
        return 4
    if a == 4:
        return 3

def main():
    samples = 100
    iterations = 100
    p_num = 2
    load_model = False
    parent_dir = os.getcwd()
    directory1 = "../../results/"
    dir_name = os.path.join(parent_dir, directory1)
    folder = '30_Nov_2021_22_43_31'
    p_file = dir_name + folder + "/p_model_epi50000"
    e_file = dir_name + folder + "/e_model_epi50000"

    # 1. Load Environment and Q-table structure
    env = TurtleBotTag(p_num)
    if load_model:
        env.load(p_file, e_file)


    # 2. Parameters of Q-leanring
    step_num = 999
    epis = 200000
    rev_list_p = []  # rewards per episode calculate
    rev_list_e = []  # rewards per episode calculate
    steps_list = []  # steps per episode
    env.RENDER_FREQ = 1000  # How often to render an episode
    env.RENDER_PLOTS = True  # whether or not to render plots
    env.SAVE_PLOTS = True  # Whether or not to save plots

    batch_size = 16

    # 3. Q-learning Algorithm
    for i in range(epis):
        # Reset environment
        s_p, s_e = env.reset()
        rAll_p = 0
        rAll_e = 0
        if i != 0 and env.epsilon > env.epsilon_min:
            env.epsilon *= env.epsilon_decay
        env.epis = i

        # The Q-Table learning algorithm
        j = 0
        while j < step_num:
            env.step_num = j
            env.render()
            j += 1

            # try move with act
            a_p, a_e = env.act(s_p, s_e)
            p_next_poses, e_next_pose = env.try_step(a_p, a_e)

            p_actions = env.decodePAction(a_p)
            new_p_actions = []
            for idx, p in enumerate(env.map.r_p):
                if p.last_pose == p.pose == p_next_poses[idx]:
                    new_p_actions.append(random.randrange(env.e_action_dim))
                else:
                    new_p_actions.append(p_actions[idx])
            # new a_p
            a_p = env.encodePAction(new_p_actions)
            # new a_e
            if env.map.r_e.pose == env.map.r_e.last_pose == e_next_pose:
                a_e = random.randrange(env.e_action_dim)

            # step
            s1_p, s1_e, r_p, r_e, d = env.step(a_p, a_e)

            env.memorize(s_p, a_p, r_p, s1_p, d, 'p')
            env.memorize(s_e, a_e, r_e, s1_e, d, 'e')
            rAll_p += r_p
            rAll_e += r_e
            s_p = s1_p
            s_e = s1_e

            if len(env.e_memory) > batch_size:
                env.replay(batch_size, 'p')
                env.replay(batch_size, 'e')
                env.update_target_model()

            if d:
                break

        rev_list_p.append(rAll_p)
        rev_list_e.append(rAll_e)
        steps_list.append(j)
        env.render()

        if i % 1000 == 0 and i != 0:
            print("p reward: {}, e reward: {}, steps: {}".format(rAll_p, rAll_e, j))
            if not os.path.exists(env.dir_name + '/model_allVision'):
                os.mkdir(env.dir_name + '/model_allVision')
            p_model_name = env.dir_name + '/model_allVision' + '/p_model_epi' + str(i)
            e_model_name = env.dir_name + '/model_allVision' + '/e_model_epi' + str(i)
            env.save(p_model_name, e_model_name)

            fname = env.dir_name
            np.savetxt(fname + "/RevListP.txt", rev_list_p)
            np.savetxt(fname + "/RevListE.txt", rev_list_e)
            np.savetxt(fname + "/StepsList.txt", steps_list)

    print("Pursuer Reward Sum on all episodes " + str(sum(rev_list_p)/epis))
    print("Evader Reward Sum on all episodes " + str(sum(rev_list_e)/epis))

    fname = env.dir_name
    fP = open(fname + "/bestPolicyStats.txt", "w+")
    fP.write("Pursuer Final Values Q-Table:\n")
    fP.write("gma = " + str(env.gamma) + "\n" )
    fP.write("step_num = " + str(step_num) + "\n")
    fP.write("epis = " + str(epis) + "\n")
    fP.close()

    np.savetxt(fname + "/RevListP.txt", rev_list_p)
    np.savetxt(fname + "/RevListE.txt", rev_list_e)
    np.savetxt(fname + "/StepsList.txt", steps_list)


if __name__ == '__main__':
    main()
