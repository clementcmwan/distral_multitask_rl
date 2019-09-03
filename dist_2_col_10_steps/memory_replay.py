import numpy as np
import torch
import torch.nn.functional as F
import random
from collections import namedtuple

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'time'))


### modified memory buffer class to accomedate 10 step rollout sampling for DQN
class ReplayMemory(object):

    def __init__(self, capacity, policy_capacity, n_step, gamma):
        self.capacity = capacity
        self.memory = []
        self.position = 0

        self.policy_capacity = policy_capacity
        self.policy_memory = []
        self.policy_position = 0

        self.n_step_buffer = []
        self.n_step = n_step
        self.gamma = gamma

    def push(self, model, target_model, alpha, beta, *args):
        """Saves a transition."""

        if len(self.policy_memory) < self.policy_capacity:
            self.policy_memory.append(None)
        self.policy_memory[self.policy_position] = Transition(*args)
        self.policy_position = (self.policy_position + 1) % self.policy_capacity

        self.n_step_buffer.append(Transition(*args))
        if len(self.n_step_buffer) < self.n_step:
            return

        # summing discounted rewards
        with torch.no_grad():
            R = 0
            for i in range(self.n_step):
                act = int(self.n_step_buffer[i][1])
                ri = self.n_step_buffer[i][-2]

                # calculating numerator of equation 8 from Teh et al.
                term = alpha*target_model(self.n_step_buffer[i][0]) + beta *model(self.n_step_buffer[i][0])
                max_term = torch.max(term)

                pi_i = F.softmax(term-max_term, dim=1)[0]
                pi_0 = F.softmax(target_model(self.n_step_buffer[i][0]), dim=1)[0]

                log_term = (alpha/beta)*torch.log(pi_0[act]) - (1/beta)*torch.log(pi_i[act])
                reg_ri = ri + log_term

                R += self.gamma**i * reg_ri

            state, action, _, _, _ = self.n_step_buffer.pop(0)

            n_th_state = self.n_step_buffer[-1][0]

            if len(self.memory) < self.capacity:
                self.memory.append(None)
            self.memory[self.position] = (state, action, R, n_th_state)
            self.position = (self.position + 1) % self.capacity

        
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def policy_sample(self, batch_size):
        return random.sample(self.policy_memory, batch_size)        

    def __len__(self):
        return len(self.memory)

    def policy_length(self):
        return len(self.policy_memory)

    def clear_memory(self):
        self.memory = []

    def get_n(self):
        return self.n_step