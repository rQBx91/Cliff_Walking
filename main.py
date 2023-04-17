import numpy as np
import random
import matplotlib.pyplot as plt
import os
from sys import platform
import time

# Constants
ALPHA = 0.1
EPSILON = 0.1
GAMMA = 1
LEN = 12
WID = 4

script_dir = ""
# File and script path for windows platform
if platform == "win64" or platform == "win32":
    script_dir = os.path.dirname(__file__) # get script execution path
# File and script path for GNU/Linux palatform
if platform == "linux" or platform == "linux2": # check for platform
    script_dir = os.getcwd() # get script execution path
        
def Main():        
    # calculate move destinatios
    action_dest = []
    for i in range(0, 12):
        action_dest.append([])
        for j in range(0, 4):
            destination = dict()
            destination[0] = [i, min(j+1,3)]
            destination[1] = [min(i+1,11), j]
            if 0 < i < 11 and j == 1:
                destination[2] = [0,0]
            else:
                destination[2] = [i, max(j - 1, 0)]
            destination[3] = [max(i-1,0), j]
            action_dest[-1].append(destination)
    action_dest[0][0][1] = [0,0]

    # calculate initial move rewards
    action_reward = np.zeros((LEN, WID, 4))
    action_reward[:, :, :] = -1.0
    action_reward[1:11, 1, 2] = -100.0
    action_reward[0, 0, 1] = -100.0

    # move the agent from location with given action
    def move_agent(x,y,a):
        goal = 0
        if x == LEN - 1 and y == 0:
            goal = 1
        if a == 0:
            y += 1
        if a == 1:
            x += 1
        if a == 2:
            y -= 1
        if a == 3:
            x -= 1

        x = max(0,x)
        x = min(LEN-1, x)
        y = max(0,y)
        y = min(WID-1, y)

        if goal == 1:
            return x,y,-1
        if x>0 and x<LEN-1 and y==0:
            return 0,0,-100
        return x,y,-1

    # decide next move with epsilon greedy
    def epsGreedyPolicy(x,y,q,eps):
        t = random.randint(0,3)
        if random.random() < eps:
            a = t
        else:
            q_max = q[x][y][0]
            a_max = 0
            for i in range(4):
                if q[x][y][i] >= q_max:
                    q_max = q[x][y][i]
                    a_max = i
            a = a_max
        return a

    # find max Q in all actions
    def findMaxQ(x,y,q):
        q_max = q[x][y][0]
        a_max = 0
        for i in range(4):
            if q[x][y][i] >= q_max:
                q_max = q[x][y][i]
                a_max = i
        a = a_max
        return a

    # SARSA algorithm implementation
    def sarsa(env):
        runs = 30
        rewards = np.zeros([500])
        for j in range(runs):
            for i in range(500):
                reward_sum = 0
                x = 0
                y = 0
                a = epsGreedyPolicy(x,y,env,EPSILON)
                while True:
                    [x_next,y_next] = action_dest[x][y][a]
                    reward = action_reward[x][y][a]
                    reward_sum += reward
                    a_next = epsGreedyPolicy(x_next,y_next,env,EPSILON)
                    env[x][y][a] += ALPHA*(reward + GAMMA*env[x_next][y_next][a_next]-env[x][y][a])
                    if x == LEN - 1 and y==0:
                        break
                    x = x_next
                    y = y_next
                    a = a_next
                rewards[i] += reward_sum
        rewards /= runs
        avg_rewards = []
        for i in range(9):
            avg_rewards.append(np.mean(rewards[:i+1]))
        for i in range(10,len(rewards)+1):
            avg_rewards.append(np.mean(rewards[i-10:i]))
        return avg_rewards

    # Q-Learning algorithm implementation
    def q_learning(env):
        runs = 30
        rewards = np.zeros([500])
        for j in range(runs):
            for i in range(500):
                reward_sum = 0
                x = 0
                y = 0
                while True:
                    a = epsGreedyPolicy(x,y,env,EPSILON)             
                    x_next, y_next,reward = move_agent(x,y,a)
                    a_next = findMaxQ(x_next,y_next,env)
                    reward_sum += reward
                    env[x][y][a] += ALPHA*(reward + GAMMA*env[x_next][y_next][a_next]-env[x][y][a])
                    if x == LEN - 1 and y==0:
                        break
                    x = x_next
                    y = y_next
                rewards[i] += reward_sum
        rewards /= runs
        avg_rewards = []
        for i in range(9):
            avg_rewards.append(np.mean(rewards[:i+1]))
        for i in range(10,len(rewards)+1):
            avg_rewards.append(np.mean(rewards[i-10:i]))
        return avg_rewards

    # find and show the optimal policy
    def showOptimalPolicy(env):
        res = ""
        for j in range(WID-1,-1,-1):
            for i in range(LEN):
                a = findMaxQ(i,j,env)
                if a == 0:
                    res += " U "
                if a == 1:
                    res += " R "
                if a == 2:
                    res += " D "
                if a == 3:
                    res += " L "
            res += "\n"
        return res

    # find and show the optimal path
    def showOptimalPath(env):
        x = 0
        y = 0
        path = np.zeros([LEN,WID]) - 1
        end = 0
        exist = np.zeros([LEN,WID])
        while (x != LEN-1 or y != 0) and end == 0:
            a = findMaxQ(x,y,env)
            path[x][y] = a
            if exist[x][y] == 1:
                end = 1
            exist[x][y] = 1
            x,y,r = move_agent(x,y,a)
        res = ''
        for j in range(WID-1,-1,-1):
            for i in range(LEN):
                if i == 0 and j == 0:
                    res += " S "
                    continue
                if i == LEN-1 and j == 0:
                    res += " G "
                    continue
                a = path[i,j]
                if a == -1:
                    res += " - "

                elif a == 0:
                    res += " U "
                elif a == 1:
                    res += " R "
                elif a == 2:
                    res += " D "
                elif a == 3:
                    res += " L "
            res += '\n'
        return res
    
    
    # run SARRSA and Q-Learning on the env
    sarsa_env = np.zeros([12,4,4])
    sarsa_rewards = sarsa(sarsa_env)
    qlearn_env = np.zeros([12,4,4])
    q_learning_rewards = q_learning(qlearn_env)

    # plot the result
    plt.plot(range(1,len(sarsa_rewards)+1), sarsa_rewards, label="SARSA")
    plt.plot(range(1,len(q_learning_rewards)+1), q_learning_rewards, label="Q-Learning")
    plt.ylim(-100,0)
    plt.xlabel('Episodes')
    plt.ylabel('Sum of rewards')
    plt.legend(loc="lower right")
    plt.title('Q-Learning vs SARSA')
    pltPath = f'{script_dir}/Q-Learning vs SARSA.png'
    plt.savefig(pltPath, dpi=200)
    print(f'saved figure to: {pltPath}')
    
    # save Optimat Policy and Optimal Path to txt file
    infoPath = f'{script_dir}/Optimal.txt'
    with open(infoPath, 'w') as file:
        file.write("SARSA Optimal Policy:\n"+ showOptimalPolicy(sarsa_env) + "\n")
        file.write("Q-learning Optimal Policy:\n"+ showOptimalPolicy(qlearn_env) + "\n")
        file.write("SARSA Optimal Path:\n"+ showOptimalPath(sarsa_env) + "\n")
        file.write("Q-learning Optimal Path:\n"+ showOptimalPath(qlearn_env) + "\n")
        print(f'save text to: {infoPath}')
    
    
if __name__ == "__main__":
    stime = time.time()
    Main()
    print("\nscript execution time: {0}\n".format(time.time() - stime) )