from enviroment import Game
import numpy as np
from agent import Agent
import os


def main():

    print("Chose one option:")
    print(" 1) play")
    print(" 2) train")
    x = input(" > ")

    if x == "1":
        print("Number of games:")
        n_games = int(input(" > "))
        print("Load path:")
        load_path = input(" > ")

        if not os.path.exists(f"models/{load_path}"):
            q_table = np.random.uniform(0.0 , 1.0, size=(50,60,42,60,2))
        else:
            q_table = np.load(f"models/{load_path}/q_table.npy")

        game = Game(q_table)
        game.play_AI(n_games)
    else:
        print("Epochs:")
        epochs = int(input(" > "))
        print("Save path:")
        save_path = input(" > ")
        print("Load path:")
        load_path = input(" > ")

        if not os.path.exists(f"models/{load_path}"):
            q_table = np.random.uniform(0.0 , 1.0, size=(50,60,42,60,2))
        else:
            q_table = np.load(f"models/{load_path}/q_table.npy")

        agent = Agent(q_table, epsilon_decay=0.5)
        agent.train(epochs=epochs, save_path=save_path) 


if __name__ == '__main__':
    main()