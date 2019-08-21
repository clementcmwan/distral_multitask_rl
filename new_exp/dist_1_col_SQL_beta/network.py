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


# use_cuda = torch.cuda.is_available()

# FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
# LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
# ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
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
    Selects whether the next action is choosen by our model or randomly
    """
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    if sample <= eps_threshold:
        return LongTensor([[random.randrange(num_actions)]])


    with torch.no_grad():
        Q = model(Variable(state).type(FloatTensor))
        
        #### FOUND ERROR: ( Q ) returns a tensor of nan at some point
        # if np.isnan( Q.sum(1).data[0]) :
        #     print("Q = ", Q)
        #     print("state = ", state)

        # pi0_a_pref = policy.forward_action_pref(Variable(state).type(FloatTensor))

        # calculate the numerator of equation 8
        # term = alpha*pi0_a_pref + beta*Q
        # max_term = torch.max(term)
        
        # get equation 8, pi_i(a_t | s_t)
        # pi_i = torch.exp(term-max_term)/(torch.exp(term-max_term).sum(1))
        pi_i = F.softmax(beta*Q,dim=1)

        # debugging for previous encounters of nan
        try:
            choice = torch.tensor([np.random.choice(num_actions, 1, p=pi_i.numpy()[0])])
        except:
            temp = model(Variable(state))
            max_term = torch.max(temp)
            print(torch.exp(temp-max_term)/(torch.exp(temp-max_term).sum(1)))
            print(pi_i)

    return choice

# opt distilled policy using equation 5
def optimize_policy(policy, optimizer, memories, batch_size,
                    num_envs, gamma, alpha, beta):
    loss = 0
    for i_env in range(num_envs):
        size_to_sample = np.minimum(batch_size, len(memories[i_env]))
        transitions = memories[i_env].policy_sample(size_to_sample)
        batch = Transition(*zip(*transitions))
        
        state_batch = Variable(torch.cat(batch.state))
        time_batch = Variable(torch.cat(batch.time))

        # this is wrong.
        # actions = np.array([action.numpy()[0][0] for action in batch.action])

        action_batch = torch.cat(batch.action)
        cur_loss = (torch.pow(Variable(Tensor([gamma])), time_batch) *
            torch.log(policy(state_batch).gather(1, action_batch))).sum()

        loss -= cur_loss

    # loss = Variable(Tensor([alpha/beta]))*loss
    optimizer.zero_grad()
    loss.backward()

    for param in policy.parameters():
        param.grad.data.clamp_(-500, 500)
    optimizer.step()


# this is the SQL part for each task specific policy
def optimize_model(policy, model, optimizer, memory, batch_size,
                    alpha, beta, gamma):
    if len(memory) < batch_size:
        return
    transitions = memory.sample(batch_size)
    # Transpose the batch (see http://stackoverflow.com/a/19343/3343043 for
    # detailed explanation).
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    non_final_mask = ByteTensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)))
    # We don't want to backprop through the expected action values and volatile
    # will save us on temporarily changing the model parameters'
    # requires_grad to False!
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
    # non_final_next_states.requires_grad = False

    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # calculate the numerator of equation 8
    # pi0_a_pref = policy.forward_action_pref(state_batch)
    term = beta*model(state_batch)
    max_term = torch.max(term, 1)[0].unsqueeze(1)

    # get equation 8, pi_i(a_t | s_t)
    pi_i = torch.exp(term-max_term)/(torch.exp(term-max_term).sum(1).unsqueeze(1))

    # reg rewards
    reward_batch = (reward_batch.unsqueeze(1) + (alpha/beta)*torch.log(policy.forward(state_batch).gather(1, action_batch))
                     - (1/beta)*torch.log(pi_i.gather(1, action_batch)))

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken
    state_action_values = model(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states, 2nd component of equation 7
    next_state_values = torch.zeros(batch_size).type(Tensor)
    next_state_values[non_final_mask] = ( torch.log(
        (torch.pow(policy.forward(non_final_next_states), alpha)
        * (torch.exp(beta * model(non_final_next_states)) + 1e-16)).sum(1)) / beta ).detach()
    
    if np.isnan(next_state_values.sum().data.numpy()):
        print('true')

    # Now, we don't want to mess up the loss with a volatile flag, so let's
    # clear it. After this, we'll just end up with a Variable that has
    # requires_grad=False
    # next_state_values.volatile = False
    # Compute the expected Q values
    expected_state_action_values = (next_state_values.unsqueeze(1) * gamma) + reward_batch

    # Compute Huber loss
    loss = F.mse_loss(state_action_values, expected_state_action_values)
    # loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in model.parameters():
        param.grad.data.clamp_(-500, 500)
    optimizer.step()
