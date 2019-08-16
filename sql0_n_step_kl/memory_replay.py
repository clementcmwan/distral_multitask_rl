import random
from collections import namedtuple
import torch
import torch.nn.functional as F

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity, n_step, gamma):
        self.capacity = capacity
        self.memory = []
        self.position = 0

        self.n_step_buffer = []
        self.n_step = n_step
        self.gamma = gamma

    def push(self, model, target_model, *args):
        """Saves a transition."""
        self.n_step_buffer.append(Transition(*args))
        if len(self.n_step_buffer) < self.n_step:
            return

        # sum of discounted rewards
        # R = sum([self.n_step_buffer[i][-1]*(self.gamma**i) for i in range(self.n_step)])

        # sum of regularized discounted rewards
        with torch.no_grad():
            R = 0
            for i in range(self.n_step):
                ri = self.n_step_buffer[i][-1]
                pi_i = F.softmax(model(self.n_step_buffer[i][0]), dim=1)
                pi_0 = F.softmax(target_model(self.n_step_buffer[i][0]), dim=1)
                log_term = torch.log(pi_i)-torch.log(pi_0)
                reg_ri = ri + (pi_i*log_term).sum()

                R += self.gamma**i * reg_ri

            if self.n_step != 1:
                state, action, _, _ = self.n_step_buffer.pop(0)
                n_th_state = self.n_step_buffer[-1][0]
            else:
                state, action, n_th_state, _ = self.n_step_buffer.pop(0)

            if len(self.memory) < self.capacity:
                self.memory.append(None)
            self.memory[self.position] = (state, action, R, n_th_state)
            self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

    def get_n(self):
        return self.n_step