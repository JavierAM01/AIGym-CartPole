import gym
import random
import time
import numpy as np 
import os
import copy
import matplotlib.pyplot as plt



class Agent:

    def __init__(self, q_table, lr=0.01, gamma=0.9, epsilon=1, epsilon_decay=0.995):

        self.env = gym.make("CartPole-v1")
        self.q_table = q_table
        
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay

    def discrete_state(self, state):
        obs_range = (2.5, 3, 0.21, 3)
        steps = (0.1,0.1,0.01,0.1)
        f = lambda i, x : 0 if x <= 0 else (x if x < obs_range[i] else obs_range[i] - steps[i] / 2)
        state2 = [f(i, obs+aux) for i, obs, aux in zip([0,1,2,3], state, obs_range)]
        index = tuple([int(obs / step) for obs, step in zip(state2, steps)])
        return index

    def train(self, epochs, save_path):

        plot_times = [0]
        plot_rewards = [0]

        previous_reward = 0
        best_mean_reward = 0
        total_reward = 0
        total_time = 0

        q_table = copy.deepcopy(self.q_table)

        for epoch in range(1,epochs+1):
            
            obs = self.env.reset()
            state = self.discrete_state(obs)
            
            done = False
            epoch_reward = 0
            t_init = time.time()

            while not done:
                
                # play
                if random.random() > self.epsilon:
                    action = np.argmax(q_table[state])
                else:
                    action = random.randint(0,1)

                obs, _, done, _ = self.env.step(action)
                
                reward = 1 if not done else -1
                epoch_reward += 1
                
                new_state = self.discrete_state(obs)

                # train
                Q_value = q_table[state + (action, )]
                if not done:
                    Q_next = np.max(q_table[new_state])
                    Q_new = (1 - self.lr) * Q_value + self.lr * (reward + self.gamma * Q_next)
                else:
                    Q_new = Q_value + self.lr * reward 
                q_table[state + (action, )] = Q_new

                state = new_state

            # update epsilon 
            if self.epsilon > 0.05:
                if epoch_reward > previous_reward and epoch > 1000:
                    self.epsilon = pow(self.epsilon_decay, epoch // 1000)

            # updates variables
            t_final = time.time()

            epoch_time = t_final - t_init
            total_time += epoch_time

            total_reward += epoch_reward
            previous_reward = epoch_reward

            # Visualize training
            if epoch % 1000 == 0:
                print("\nEpoch:", epoch, f"(e = {self.epsilon})")

                mean_reward = total_reward / 1000
                print("Mean reward:", mean_reward)
                total_reward = 0
                plot_rewards.append(mean_reward)
                
                mean_times = total_time / 1000
                print("Mean times:", mean_times)
                total_time = 0
                plot_times.append(mean_times)

                if plot_rewards[-1] > best_mean_reward:
                    best_mean_reward = plot_rewards[-1]
                    print("Update q_table!")
                    self.q_table = q_table
                q_table = copy.deepcopy(self.q_table)

        # plotings & savings
        path = f"models/{save_path}"
        if not os.path.exists(path):
            os.mkdir(path)

        x = range(len(plot_times))
        
        plt.clf()
        plt.plot(x, plot_rewards)
        plt.title("Mean rewards")
        plt.xlabel("epoch")
        plt.savefig(f"{path}/rewards.png")
        plt.show()
        
        plt.clf()
        plt.plot(x, plot_times)
        plt.title("Mean times")
        plt.xlabel("epoch")
        plt.savefig(f"{path}/times.png")
        plt.show()

        np.save(f"{path}/q_table.npy", self.q_table)

        self.env.close()