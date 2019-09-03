import math
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from memory_replay import Transition
from itertools import count
from torch.distributions import Categorical


FloatTensor = torch.FloatTensor
LongTensor = torch.LongTensor
ByteTensor = torch.ByteTensor
Tensor = FloatTensor

class DQN(nn.Module):
    """
    DQN for each task specific agent.
    """
    def __init__(self, input_size, num_actions):
        super(DQN, self).__init__()
        self.linear1 = nn.Linear(input_size, 64)
        self.linear2 = nn.Linear(64, 128)
        self.linear3 = nn.Linear(128, 64)
        self.head = nn.Linear(64, num_actions)

    def forward(self, x):
        # returns Q(s,a) for a task specific policy
        x = F.leaky_relu(self.linear1(x))
        x = F.leaky_relu(self.linear2(x))
        x = F.leaky_relu(self.linear3(x))
        return self.head(x)

class PolicyNetwork(nn.Module):
    """
    DQN for the distilled policy.
    """
    def __init__(self, input_size, num_actions):
        super(PolicyNetwork, self).__init__()
        self.linear1 = nn.Linear(input_size, 64)
        self.linear2 = nn.Linear(64, 128)
        self.linear3 = nn.Linear(128, 64)
        self.head = nn.Linear(64, num_actions)

    def forward(self, x):
        # returns pi(a|s)
        x = F.leaky_relu(self.linear1(x))
        x = F.leaky_relu(self.linear2(x))
        x = F.leaky_relu(self.linear3(x))
        return F.softmax(self.head(x), dim=1)

    def forward_action_pref(self, x):
        # returns action preference of pi_0
        x = F.leaky_relu(self.linear1(x))
        x = F.leaky_relu(self.linear2(x))
        x = F.leaky_relu(self.linear3(x))
        return self.head(x)

def select_action(state, policy, model, num_actions,
                    EPS_START, EPS_END, EPS_DECAY, steps_done, alpha, beta):
    """
    Selects whether the next action is chosen by our model or randomly
    """
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    if sample <= eps_threshold:
        return LongTensor([[random.randrange(num_actions)]])


    with torch.no_grad():
        Q = model(Variable(state).type(FloatTensor))

        # get action pref from pi_0
        pi0_a_pref = policy.forward_action_pref(Variable(state).type(FloatTensor))

        # calculate the numerator of equation 8 from Teh et al.
        term = alpha*pi0_a_pref + beta*Q
        max_term = torch.max(term)
        
        # get equation 8, pi_i(a_t | s_t) from Teh et al.
        pi_i = torch.exp(term-max_term)/(torch.exp(term-max_term).sum(1))

        # pick the action according to pi_i given current state
        action = torch.tensor([np.random.choice(num_actions, 1, p=pi_i.numpy()[0])])

    return action

def optimize_policy(policy, optimizer, memories, batch_size,
                    num_envs, gamma, alpha, beta):
    '''
    Optimize the distilled policy via stochastic gradient descent
    by sampling from equation 5 from Teh et al.
    '''

    loss = 0
    for i_env in range(num_envs):
        # determine sample size
        size_to_sample = np.minimum(batch_size, len(memories[i_env]))

        # extract sample
        transitions = memories[i_env].policy_sample(size_to_sample)

        # format sampled batch
        batch = Transition(*zip(*transitions))

        # format state, time and action batch
        state_batch = torch.cat(batch.state)
        time_batch = torch.cat(batch.time)
        action_batch = torch.cat(batch.action)

        # add to current loss according to equation 5 from Teh et al.
        cur_loss = (torch.pow(Variable(Tensor([gamma])), time_batch) *
            torch.log(policy(state_batch).gather(1, action_batch))).sum()

        loss -= cur_loss

    loss = (alpha/beta) * loss

    optimizer.zero_grad()
    loss.backward()

    for param in policy.parameters():
        param.grad.data.clamp_(-500, 500)
    optimizer.step()


# this is the SQL part for each task specific policy
def optimize_model(policy, model, optimizer, memory, batch_size,
                    alpha, beta, gamma):
    '''
    Optimize w.r.t. task-specific policies via MSE and TD-learning
    '''

    if len(memory) < batch_size:
        return
    transitions = memory.sample(batch_size)
    # format transition tuple
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    non_final_mask = ByteTensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)))

    # extract all none terminal next states
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

    # format state, action and rewards
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # calculate the numerator of equation 8 from Teh et al.
    pi0_a_pref = policy.forward_action_pref(state_batch)
    term = alpha*pi0_a_pref + beta*model(state_batch)
    max_term = torch.max(term, 1)[0].unsqueeze(1)

    # get pi_i(a_t | s_t) via equation 8 from Teh et al.
    pi_i = torch.exp(term-max_term)/(torch.exp(term-max_term).sum(1).unsqueeze(1))

    # compute regularized rewards as defined in Teh et al.
    reward_batch = (reward_batch.unsqueeze(1) + (alpha/beta)*torch.log(policy.forward(state_batch).gather(1, action_batch))
                     - (1/beta)*torch.log(pi_i.gather(1, action_batch)))

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken
    state_action_values = model(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states, 2nd component of equation 7 from Teh et al.
    next_state_values = torch.zeros(batch_size).type(Tensor)
    next_state_values[non_final_mask] = ( torch.log(
        (torch.pow(policy.forward(non_final_next_states), alpha)
        * (torch.exp(beta * model(non_final_next_states)) + 1e-16)).sum(1)) / beta ).detach()

    # Compute the expected Q values
    expected_state_action_values = (next_state_values.unsqueeze(1) * gamma) + reward_batch

    # Compute loss
    loss = F.mse_loss(state_action_values, expected_state_action_values)

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in model.parameters():
        param.grad.data.clamp_(-500, 500)
    optimizer.step()
