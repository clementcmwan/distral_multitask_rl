import matplotlib
import matplotlib.pyplot as plt
from itertools import count
import torch.optim as optim
import torch
import math
import numpy as np
from memory_replay import ReplayMemory, Transition
from network import DQN, select_action, optimize_model, Tensor, optimize_policy, PolicyNetwork
# from network import DQN, select_action, optimize_model, optimize_policy, PolicyNetwork
import sys
sys.path.append('../../envs/')
from gridworld_env import GridworldEnv

def trainD(file_name="Distral_1col", list_of_envs=[GridworldEnv(5),
            GridworldEnv(4), GridworldEnv(6)], batch_size=128, gamma=0.999, alpha=0.8,
            beta=5, eps_start=0.9, eps_end=0.05, eps_decay=5,
            is_plot=False, num_episodes=200,
            max_num_steps_per_episode=1000, learning_rate=0.001,
            memory_replay_size=10000, memory_policy_size=1000):
    """
    Soft Q-learning training routine. Retuns rewards and durations logs.
    Plot environment screen
    """
    num_actions = list_of_envs[0].action_space.n
    input_size = list_of_envs[0].observation_space.shape[0]
    num_envs = len(list_of_envs)
    policy = PolicyNetwork(input_size, num_actions)
    models = [DQN(input_size,num_actions) for _ in range(0, num_envs)]   ### Add torch.nn.ModuleList (?)
    memories = [ReplayMemory(memory_replay_size, memory_policy_size) for _ in range(0, num_envs)]

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        policy.cuda()
        for model in models:
            model.cuda()

    optimizers = [optim.Adam(model.parameters(), lr=learning_rate)
                    for model in models]
    policy_optimizer = optim.Adam(policy.parameters(), lr=learning_rate)
    # optimizer = optim.RMSprop(model.parameters(), )

    episode_durations = [[] for _ in range(num_envs)]
    episode_rewards = [[] for _ in range(num_envs)]

    steps_done = np.zeros(num_envs)
    episodes_done = np.zeros(num_envs)
    current_time = np.zeros(num_envs)

    # Initialize environments
    states = []
    for env in list_of_envs:
        states.append(torch.from_numpy( env.reset() ).type(torch.FloatTensor).view(-1,input_size))

    while np.min(episodes_done) < num_episodes:
        # TODO: add max_num_steps_per_episode

        # Optimization is given by alterating minimization scheme:
        #   1. do the step for each env
        #   2. do one optimization step for each env using "soft-q-learning".
        #   3. do one optimization step for the policy

        for i_env, env in enumerate(list_of_envs):
        
            # select an action
            action = select_action(states[i_env], policy, models[i_env], num_actions,
                                    eps_start, eps_end, eps_decay,
                                    episodes_done[i_env], alpha, beta)

            steps_done[i_env] += 1
            current_time[i_env] += 1
            next_state_tmp, reward, done, _ = env.step(action[0,0])
            reward = Tensor([reward])

            # Observe new state
            next_state = torch.from_numpy( next_state_tmp ).type(torch.FloatTensor).view(-1,input_size)

            if done:
                next_state = None

            # Store the transition in memory
            time = Tensor([current_time[i_env]])
            memories[i_env].push(states[i_env], action, next_state, reward, time)

            # Perform one step of the optimization (on the target network)
            optimize_model(policy, models[i_env], optimizers[i_env],
                            memories[i_env], batch_size, alpha, beta, gamma)

            # Update state
            states[i_env] = next_state

            # Check if agent reached target
            if done or current_time[i_env] >= max_num_steps_per_episode:
                if episodes_done[i_env] <= num_episodes:
                    print("ENV:", i_env, "iter:", episodes_done[i_env],
                        "\treward:{0:.2f}".format(env.episode_total_reward),
                        "\tit:", current_time[i_env], "\texp_factor:", eps_end +
                        (eps_start - eps_end) * math.exp(-1. * episodes_done[i_env] / eps_decay))

                episode_rewards[i_env].append(env.episode_total_reward)
                episodes_done[i_env] += 1
                episode_durations[i_env].append(current_time[i_env])
                current_time[i_env] = 0

                states[i_env] = torch.from_numpy( env.reset() ).type(torch.FloatTensor).view(-1,input_size)


        optimize_policy(policy, policy_optimizer, memories, batch_size,
                    num_envs, gamma, alpha, beta)

    print('Complete')
    env.render(close=True)
    env.close()
    if is_plot:
        plt.ioff()
        plt.show()

    ## Store Results

    np.save(file_name + '-distral-2col-rewards', episode_rewards)
    np.save(file_name + '-distral-2col-durations', episode_durations)

    return models, policy, episode_rewards, episode_durations

if __name__ == '__main__':
    # trainD(list_of_envs=[GridworldEnv(4),GridworldEnv(5),GridworldEnv(6),GridworldEnv(7),GridworldEnv(8)], learning_rate=0.001, max_num_steps_per_episode=100, num_episodes=1000)
    # trainD(list_of_envs=[GridworldEnv(4),GridworldEnv(5),GridworldEnv(6),GridworldEnv(7),GridworldEnv(8)], learning_rate=0.001)
    trainD(list_of_envs=[GridworldEnv(4),GridworldEnv(5)], learning_rate=0.001, beta=4)