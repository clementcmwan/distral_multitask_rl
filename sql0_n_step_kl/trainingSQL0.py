import matplotlib
import matplotlib.pyplot as plt
from itertools import count
import torch.optim as optim
import torch
import math
import numpy as np
from memory_replay import ReplayMemory, Transition
from network import DQN, select_action, optimize_model, Tensor
import sys
from IPython.display import clear_output
sys.path.append('../')
sys.path.append('../envs/')
from gridworld_env import GridworldEnv
from utils import plot_rewards, plot_durations, plot_state, get_screen

def trainSQL0(file_name="SQL0", env=GridworldEnv(1), batch_size=128,
            gamma=0.999, beta=5, eps_start=0.9, eps_end=0.05, eps_decay=1000,
            is_plot=False, num_episodes=200, max_num_steps_per_episode=1000,
            learning_rate=0.0001, memory_replay_size=10000, n_step=10, target_update=10):
    """
    Soft Q-learning training routine when observation vector is input
    Retuns rewards and durations logs.
    """

    num_actions = env.action_space.n
    input_size = env.observation_space.shape[0]
    model = DQN(input_size, num_actions)
    target_model = DQN(input_size, num_actions)
    target_model.load_state_dict(model.state_dict())
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # optimizer = optim.RMSprop(model.parameters(), )

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        model.cuda()

    memory = ReplayMemory(memory_replay_size, n_step, gamma)

    episode_durations = []
    mean_durations = []
    episode_rewards = []
    mean_rewards = []

    steps_done, t = 0, 0

    for i_episode in range(num_episodes):
        if i_episode % 20 == 0:
            clear_output()
        if i_episode != 0:
            print("Cur episode:", i_episode, "steps done:", episode_durations[-1],
                    "exploration factor:", eps_end + (eps_start - eps_end) * \
                    math.exp(-1. * steps_done / eps_decay), "reward:", env.episode_total_reward)
        # Initialize the environment and state
        state = torch.from_numpy( env.reset() ).type(torch.FloatTensor).view(-1,input_size)

        for t in count():
            # Select and perform an action
            action = select_action(state, model, num_actions,
                                    eps_start, eps_end, eps_decay, steps_done)
            next_state_tmp, reward, done, _ = env.step(action[0, 0])
            reward = Tensor([reward])

            # Observe new state
            next_state = torch.from_numpy( next_state_tmp ).type(torch.FloatTensor).view(-1,input_size)

            if done:
                next_state = None

            # Store the transition in memory
            memory.push(model, target_model, state, action, next_state, reward)

            # Move to the next state
            state = next_state
            # plot_state(state)
            # env.render()

            # Perform one step of the optimization (on the target network)
            optimize_model(model, target_model, optimizer, memory, batch_size, gamma, beta)  #### Difference w.r.t DQN
            if done or t + 1 >= max_num_steps_per_episode:
                episode_durations.append(t + 1)
                episode_rewards.append(env.episode_total_reward)  ##### Modify for OpenAI envs such as CartPole
                if is_plot:
                    plot_durations(episode_durations, mean_durations)
                    plot_rewards(episode_rewards, mean_rewards)
                steps_done += 1
                break
        if i_episode % target_update == 0 and i_episode != 0:
            target_model.load_state_dict(model.state_dict())

    print('Complete')
    env.render(close=True)
    env.close()
    if is_plot:
        plt.ioff()
        plt.show()

    ## Store Results
    np.save(file_name + '-sql0-rewards', episode_rewards)
    np.save(file_name + '-sql0-durations', episode_durations)

    return model, episode_rewards, episode_durations


if __name__ == '__main__':
    # trainSQL0(env=GridworldEnv(4), learning_rate=0.00001, max_num_steps_per_episode=100, num_episodes=1000)
    trainSQL0(env=GridworldEnv(8), learning_rate=0.001, max_num_steps_per_episode=1000, num_episodes=200, n_step=1)