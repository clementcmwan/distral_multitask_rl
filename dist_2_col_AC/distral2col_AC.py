import argparse
import gym
import numpy as np
from itertools import count
from collections import namedtuple
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.distributions import Categorical


parser = argparse.ArgumentParser(description='PyTorch distral example')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=512, metavar='N',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=5, metavar='N',
                    help='interval between training status logs (default: 5)')
args = parser.parse_args()

import sys
sys.path.append('envs/')
from gridworld_env import GridworldEnv

class Policy(nn.Module):

    def __init__(self, input_size, num_actions ):

        super(Policy, self).__init__()
        # define network to output state-action values
        self.val1 = nn.Linear(input_size, 64)
        self.val2 = nn.Linear(64, 128)
        self.val3 = nn.Linear(128, 64)
        self.value_head = nn.Linear(64, num_actions)

        self.saved_actions = [] 
        self.rewards = [] 
        self.state_action = []
        self.pi_prob = []

    def forward(self, x):
        val = F.relu(self.val1(x))
        val = F.relu(self.val2(val))
        val = F.relu(self.val3(val))
        value_est = self.value_head(val)
        return value_est

class Distilled(nn.Module):

    def __init__(self, input_size, num_actions, num_tasks):

        super(Distilled, self).__init__()
        self.lienar1 = nn.Linear(input_size, 64)
        self.lienar2 = nn.Linear(64, 128)
        self.lienar3 = nn.Linear(128, 128)
        self.lienar4 = nn.Linear(128, 64)
        self.action_head = nn.Linear(64, num_actions) 

        self.saved_actions = [ [] for _ in range(num_tasks) ] 
        self.action_pref = [ [] for _ in range(num_tasks) ]
        self.pi_prob = [ [] for _ in range(num_tasks) ]

    def forward(self, x):
        x = F.relu(self.lienar1(x))
        x = F.relu(self.lienar2(x))
        x = F.relu(self.lienar3(x))
        x = F.relu(self.lienar4(x))

        action_scores = F.softmax(self.action_head(x), dim=-1) 
        action_pref = self.action_head(x)
        return action_scores, action_pref


def select_action(state, policy, distilled, task_id, alpha, beta):

    # Format the state
    state = torch.from_numpy(state).float()

    # Run distilled policy
    probs0, action_pref_0 = distilled.forward(Variable(state))
    distilled.action_pref[task_id].append(action_pref_0)
    distilled.pi_prob[task_id].append(probs0)

    # Run the policy
    Q = policy.forward(Variable(state))

    # calculate the numerator of equation 8
    term = alpha*action_pref_0 + beta*Q
    max_term = torch.max(term)

    # get equation 8, pi_i(a_t | s_t)
    pi_i = torch.exp(term-max_term)/(torch.exp(term-max_term).sum())
    policy.pi_prob.append(pi_i)

    # Obtain the most probable action for the policy
    m = Categorical(pi_i)
    action =  m.sample() 
    policy.saved_actions.append( m.log_prob(action))
    policy.state_action.append([state.numpy(),action.numpy()])


    # Obtain the most probably action for the distilled policy
    m = Categorical(probs0)
    # action_tmp =  m.sample() 
    distilled.saved_actions[task_id].append( m.log_prob(action) )

    # Return the most probable action for the policy
    return action


# actor critic framework for updating the task policy and q values for current task
def task_specific_update(policy, distilled, opt_policy, alpha, beta, gamma, final_state_value, task_id):

    task_policy_loss = []

    # Give format
    alpha = torch.Tensor([alpha])
    beta = torch.Tensor([beta])
    gamma = torch.Tensor([gamma])

    rewards = policy.rewards
    policy_actions = policy.saved_actions
    distill_actions = distilled.saved_actions[task_id]

    pi_probs = torch.stack(policy.pi_prob)
    p0_probs = torch.stack(distilled.pi_prob[task_id])

    # calculate MSE/huber loss for state value estimates (1 step TD)
    reg_rewards = []

    # reformating into tensors
    states, actions = np.asarray(policy.state_action)[:,0], np.asarray(policy.state_action)[:,1]
    states, actions = np.array([*states]), np.array([*actions]).reshape(-1,1)

    q_thetas = policy(torch.Tensor(states))

    # q_thetas = q_thetas.gather(1,torch.tensor(actions))
    v_thetas = torch.log((torch.pow(p0_probs,alpha) * torch.exp(beta*q_thetas)).sum(1)) / beta # from equation 55 bekerley paper
    # v_thetas = torch.log((p0_probs* torch.exp(beta*q_thetas)).sum(1)) / beta

    q_values = np.zeros(len(states))
    v_val = final_state_value

    # n step td learning, n = 10
    reg_rewards = []
    n = 1
    gammas = [gamma**i for i in range(n)]
    for t in range(len(rewards)):
        reg_rewards.append(rewards[t] + (alpha/beta)*distill_actions[t] - (1./beta)*policy_actions[t])

    for t in reversed(range(len(rewards)-n)):
        reg_reward = [g*rwd for g,rwd in zip(gammas,reg_rewards[t:t+n])]
        qval = np.sum(reg_reward) + gamma**n * v_val
        q_values[t] = qval
        v_val = v_thetas[t]

    q_values = torch.tensor(q_values).float().unsqueeze(1).detach()
    # q_values = (q_values - q_values.mean())/(q_values.std() + 1e-15)

    # MSE for 1 step TD
    # critic_loss = F.mse_loss(q_values, v_thetas.unsqueeze(1))
    critic_loss = F.smooth_l1_loss(q_values, v_thetas.unsqueeze(1))

    # get losses for current task, from equation 9
    for t, (log_prob_i) in enumerate(policy_actions): 
        advantage = (q_values[t] - v_thetas[t]).detach()
        task_policy_loss.append(log_prob_i * advantage)

    opt_policy.zero_grad()

    loss = -(torch.stack(task_policy_loss).mean()) + critic_loss

    loss.backward(retain_graph=True)

    for param in policy.parameters():
        if param.grad is not None:
            param.grad.data.clamp_(-1, 1)

    opt_policy.step()

    return torch.stack(task_policy_loss).sum()


# policy gradient for distilled policy
def finish_episode(task_specific_loss, policies, distilled, opt_distilled, alpha, beta, gamma):

    # Give format
    alpha = torch.Tensor([alpha])
    beta = torch.Tensor([beta])
    gamma = torch.Tensor([gamma])

    mismatch_loss = []

    for task_id, policy in enumerate(policies):

        mismatch_loss_i = []
        
        # Retrive distilled policy actions and action prefs
        distill_probs = distilled.pi_prob[task_id]
        distill_action_prefs = distilled.action_pref[task_id]

        # Retrieve policy actions
        policy_probs = policy.pi_prob

        # get losses for second part of equation 10
        for t, (prob_i, prob_0, action_pref_0) in enumerate(zip(policy_probs, distill_probs, distill_action_prefs)): 
            discount = gamma**t
            sum_term = ((prob_i - prob_0) * action_pref_0).sum()

            mismatch_loss_i.append(discount * sum_term)

        mismatch_loss.append(torch.stack(mismatch_loss_i).sum())        

    # Perform optimization step
    opt_distilled.zero_grad()

    loss =  -(torch.stack(task_specific_loss).sum() + (alpha/beta) * torch.stack(mismatch_loss).sum())

    # print(torch.stack(task_specific_loss).sum())
    # print((alpha/beta) * torch.stack(mismatch_loss).sum())

    loss.backward(retain_graph=True)

    # for param in distilled.parameters():
    #     param.grad.data.clamp_(-500, 500)

    opt_distilled.step()

    #Clean memory
    for ind, policy in enumerate(policies):
        del policy.rewards[:]
        del policy.saved_actions[:]
        del policy.state_action[:]
        del policy.pi_prob[:]
        del distilled.saved_actions[ind][:]
        del distilled.action_pref[ind][:]
        del distilled.pi_prob[ind][:]


def trainDistral( file_name="Distral_1col", list_of_envs=[GridworldEnv(5), GridworldEnv(4)], batch_size=128, gamma=0.95, alpha=0.8,
            beta=5, num_episodes=200,
            max_num_steps_per_episode=1000, learning_rate=0.001):

    # Specify Environment conditions
    input_size = list_of_envs[0].observation_space.shape[0]
    num_actions = list_of_envs[0].action_space.n
    tasks = len(list_of_envs)

    # Define our set of policies, including distilled one
    models = torch.nn.ModuleList( [Policy(input_size, num_actions) for _ in range(tasks)] )
    distilled = Distilled(input_size, num_actions, tasks)
    optimizers = [optim.Adam(model.parameters(), lr=learning_rate) for model in models]
    opt_distilled = optim.Adam(distilled.parameters(), lr=learning_rate)

    # Store the total rewards
    episode_rewards = [ [] for i in range(num_episodes) ]
    episode_duration = [ [] for i in range(num_episodes) ]

    for i_episode in range(num_episodes):
        task_specific_losses = []

        # For each one of the envs
        for i_env, env in enumerate(list_of_envs):

            #Initialize state of envs
            state = env.reset()

            #Store total reward per environment per episode
            total_reward = 0

            # Store duration of each episode per env
            duration = 0
            
            for t in range(max_num_steps_per_episode):

                # Run our policy
                action = select_action(state, models[i_env], distilled, i_env, alpha, beta)

                next_state, reward, done, _ = env.step(action.item())
                models[i_env].rewards.append(reward)
                total_reward += reward
                duration += 1

                if done:
                    break

                #Update state
                state = next_state

            episode_rewards[i_episode].append(total_reward) 
            episode_duration[i_episode].append(duration)

            # get the value estimate of the final state according to equation 7 from distral paper
            next_state = torch.from_numpy(np.asarray(next_state)).float()
            Q_temp = models[i_env](next_state)
            pi_0_temp, _ = distilled(next_state)
            final_state_value = torch.log((pi_0_temp *torch.exp(beta*Q_temp)).sum()) / beta            

            if done:
                final_state_value = 0

            # Distill for each environment
            task_specific_losses.append(task_specific_update(models[i_env], distilled,
                                                             optimizers[i_env], alpha,
                                                             beta, gamma, final_state_value,
                                                             i_env))

        finish_episode(task_specific_losses, models, distilled, opt_distilled, alpha, beta, gamma)

        # if i_episode % args.log_interval == 0:
        for i in range(tasks):
            print('Episode: {}\tEnv: {}\tDuration: {}\tTotal Reward: {:.2f}'.format(
                i_episode, i, episode_duration[i_episode][i], episode_rewards[i_episode][i]))


    np.save(file_name + '-distral0-rewards' , episode_rewards)
    np.save(file_name + '-distral0-durations' , episode_duration)

    print('Completed')

if __name__ == '__main__':
    # trainDistral(list_of_envs=[GridworldEnv(4), GridworldEnv(5), GridworldEnv(6), GridworldEnv(7), GridworldEnv(8)], learning_rate=0.0001, num_episodes=200)
    trainDistral(list_of_envs=[GridworldEnv(4), GridworldEnv(5)], learning_rate=0.0001, num_episodes=200)
