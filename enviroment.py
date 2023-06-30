import gym
import time
import torch as T
import numpy as np 

from agent import Agent


class Game:

    def __init__(self, q_table):
        self.env = gym.make("CartPole-v1")
        self.q_table = q_table

    def play(self):
        self.env.reset()
        action = self.env.action_space.sample()  # take a random action
        for _ in range(50):
            obs, reward, done, _ = self.env.step(action)
            self.env.render()
            action = 1 if obs[2] > 0 else 0
            time.sleep(0.1)
            print(obs)
            print(reward)
            if done:
                print("Done!\n")
                self.env.reset()
        self.env.close()

    def play_AI(self,n_games=1):
        agent = Agent(self.q_table)
        obs = self.env.reset()
        state = agent.discrete_state(obs)
        total_reward = 0
        game = 1
        while game < n_games:
            action = np.argmax(self.q_table[state])
            obs, _, done, _ = self.env.step(action)
            state = agent.discrete_state(obs)
            total_reward += 1
            self.env.render()
            if done:
                print(f"\n[{game}] Reward: {total_reward}")
                game += 1
                self.env.reset()
                total_reward = 0
        self.env.close()