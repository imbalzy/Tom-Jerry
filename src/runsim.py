import numpy as np
import matplotlib.pyplot as plt
import gym
from map import Map
from robot import Robot
from qlearning import TurtleBotTag
import datetime


def main():
    samples = 100
    iterations = 100
    p_num = 2

    # 1. Load Environment and Q-table structure
    env = TurtleBotTag(p_num)

    # 2. Parameters of Q-leanring
    step_num = 999
    epis = 200000
    rev_list_p = []  # rewards per episode calculate
    rev_list_e = []  # rewards per episode calculate
    steps_list = []  # steps per episode
    env.RENDER_FREQ = 50000  # How often to render an episode
    env.RENDER_PLOTS = True  # whether or not to render plots
    env.SAVE_PLOTS = True  # Whether or not to save plots

    batch_size = 512

    # 3. Q-learning Algorithm
    for i in range(epis):
        # Reset environment
        s_p, s_e = env.reset()
        rAll_p = 0
        rAll_e = 0
        if i != 0:
            epsilon = 1/np.floor(i/epis/20)
        env.epis = i

        # The Q-Table learning algorithm
        j = 0
        while j < step_num:
            env.step_num = j
            env.render()
            j += 1

            # Choose action from Q table based on epsilon-greedy
            a_p, a_e = env.act(s_p, s_e)

            s1_p, s1_e, r_p, r_e, d = env.step(a_p, a_e)

            env.memorize(s_p, a_p, r_p, s1_p, d, 'p')
            env.memorize(s_e, a_e, r_e, s1_e, d, 'e')
            rAll_p += r_p
            rAll_e += r_e
            s_p = s1_p
            s_e = s1_e

            if d:
                break

            if len(env.e_memory) == batch_size:
                env.replay(batch_size, 'p')
                env.replay(batch_size, 'e')

        env.update_target_model()
        rev_list_p.append(rAll_p)
        rev_list_e.append(rAll_e)
        steps_list.append(j)
        env.render()

    # print("Pursuer Reward Sum on all episodes " + str(sum(rev_list_p)/epis))
    # print("Evader Reward Sum on all episodes " + str(sum(rev_list_e)/epis))
    # print("Pursuer Final Values Q-Table:\n", Q_p)
    # print("Evader Final Values Q-Table:\n", Q_e)
    #
    # fname = env.dir_name
    # fP = open(fname + "/bestPolicyStats.txt", "w+")
    # fP.write("Pursuer Final Values Q-Table:\n")
    # fP.write("eta = " + str(eta) + "\n")
    # fP.write("gma = " + str(gma) + "\n" )
    # fP.write("step_num = " + str(step_num) + "\n")
    # fP.write("epis = " + str(epis) + "\n")
    # fP.close()
    #
    # np.save(fname + "/bestPolicyQTableP", Q_p)
    # np.save(fname + "/bestPolicyQTableE", Q_e)
    # np.savetxt(fname + "/RevListP.txt", rev_list_p)
    # np.savetxt(fname + "/RevListE.txt", rev_list_e)
    # np.savetxt(fname + "/StepsList.txt", steps_list)


if __name__ == '__main__':
    main()
