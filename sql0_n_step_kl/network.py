import math
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from memory_replay import Transition

use_cuda = torch.cuda.is_available()

FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor

class DQN(nn.Module):
    """
    Deep neural network with represents an agent.
    """
    def __init__(self, input_size, num_actions):
        super(DQN, self).__init__()
        self.linear1 = nn.Linear(input_size, 64)
        self.linear2 = nn.Linear(64, 128)
        self.linear3 = nn.Linear(128, 128)
        self.linear4 = nn.Linear(128, 64)
        self.head = nn.Linear(64, num_actions)

    def forward(self, x):
        # returns Q(a,s)
        x = F.leaky_relu(self.linear1(x))
        x = F.leaky_relu(self.linear2(x))
        x = F.leaky_relu(self.linear3(x))
        x = F.leaky_relu(self.linear4(x))
        return self.head(x)

def select_action(state, model, num_actions,
                    EPS_START, EPS_END, EPS_DECAY, steps_done):
    """
    Selects whether the next action is choosen by our model or randomly
    """
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    if sample > eps_threshold:
        with torch.no_grad():
            return model(state).type(FloatTensor).data.max(1)[1].view(1, 1)
    else:
        return LongTensor([[random.randrange(num_actions)]])


def optimize_model(model, target_model, optimizer, memory, BATCH_SIZE, GAMMA, BETA):
    global last_sync
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    state_batch, action_batch, dis_rwd_batch, n_state_batch = zip(*transitions)

    # Compute a mask of non-final states and concatenate the batch elements
    non_final_mask = ByteTensor(tuple(map(lambda s: s is not None,
                                         n_state_batch)))
    # We don't want to backprop through the expected action values and volatile
    # will save us on temporarily changing the model parameters'
    # requires_grad to False!
    non_final_next_states = torch.cat([s for s in n_state_batch if s is not None])
    non_final_next_states_action = torch.cat([action_batch[ind] for ind, s in enumerate(n_state_batch) if s is not None])

    state_batch = torch.cat(state_batch)
    action_batch = torch.cat(action_batch)
    reward_batch = torch.cat(dis_rwd_batch)

    assert math.isnan(reward_batch.sum()) != True
    assert np.isinf(reward_batch.sum().detach()) != True


    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken
    state_action_values = model(state_batch).gather(1, action_batch)

    assert math.isnan(state_action_values.sum()) != True
    assert np.isinf(state_action_values.sum().detach()) != True

    # Compute V(s_{t+1}) for all next states.
    next_state_values = torch.zeros(BATCH_SIZE).type(Tensor)

    # get pi(a|s) as in the old policy network
    term = target_model(non_final_next_states)
    max_term = term.max(1)[0].unsqueeze(1)
    pi_prob = torch.exp(term + max_term) / torch.exp(term + max_term).sum(1).unsqueeze(1)
    pi_prob = pi_prob.gather(1, non_final_next_states_action)

    expected_values_V = pi_prob * torch.exp(BETA * target_model(non_final_next_states)).sum(1).unsqueeze(1)

    next_state_values[non_final_mask] = (torch.log( expected_values_V.sum(1) ) / BETA).detach()
    next_state_values[non_final_mask] = (next_state_values[non_final_mask] - next_state_values[non_final_mask].mean()) - next_state_values[non_final_mask].std()

    # Now, we don't want to mess up the loss with a volatile flag, so let's
    # clear it. After this, we'll just end up with a Variable that has
    # requires_grad=False
    # next_state_values.volatile = False
    # Compute the expected Q values

    assert math.isnan(next_state_values.sum()) != True
    assert np.isinf(next_state_values.sum().detach()) != True

    expected_state_action_values = (next_state_values * GAMMA**memory.get_n()) + reward_batch

    # Compute Huber loss
    loss = F.mse_loss(state_action_values, expected_state_action_values.unsqueeze(1))
    # loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # for param in model.parameters():
    #     param.grad.data.clamp_(-500, 500)
    optimizer.step()
