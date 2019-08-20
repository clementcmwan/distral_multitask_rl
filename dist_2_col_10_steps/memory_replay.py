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

        # sum of discounted rewards
        # R = sum([self.n_step_buffer[i][-2]*(self.gamma**i) for i in range(self.n_step)])

        with torch.no_grad():
            R = 0
            for i in range(self.n_step):
                act = int(self.n_step_buffer[i][1])
                ri = self.n_step_buffer[i][-2]

                term = alpha*target_model(self.n_step_buffer[i][0]) + beta *model(self.n_step_buffer[i][0])
                max_term = torch.max(term)
                # pi_i = torch.exp(term-max_term)/(torch.exp(term-max_term).sum(1))
                pi_i = F.softmax(term-max_term, dim=1)[0]
                pi_0 = F.softmax(target_model(self.n_step_buffer[i][0]), dim=1)[0]

                log_term = (alpha/beta)*torch.log(pi_0[act]) - (1/beta)*torch.log(pi_i[act])
                # print(f"1st:{torch.log(pi_0[act])}, 2nd:{torch.log(pi_i[act])}")
                reg_ri = ri + log_term

                R += self.gamma**i * reg_ri

            # print(R)
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

    # def get_n_step_rollout(self, indx_batch, gamma):

    #     # returns sum of discounted rewards after rolling out for n steps for each sample

    #     # get discounts from 2nd to 9th step
    #     gammas = [gamma**i for i in range(1, self.n_step-1)]

    #     # indx_batch are the time step/ indx of the currently sampled batch
    #     # get the rewards of the n+1 th time step
    #     rewards = []
    #     for i in range(1, self.n_step-1):
    #         indx_batch = indx_batch + 1
    #         cur_rwd_batch = torch.tensor([self.memory[cur_indx][-2] for cur_indx in indx_batch]) 
    #         rewards.append(cur_rwd_batch)

    #     rewards = torch.stack(rewards).numpy()

    #     # multiply the rewards with the correpsonding discount rates
    #     dicounted_rewards = np.asarray(gammas).reshape(-1,1)*rewards
    #     # sum the n-step rewards for each sample trajectory
    #     dis_rwd_sum = np.sum(dicounted_rewards,axis =0).reshape(-1,1)

    #     return torch.tensor(dis_rwd_sum)

    # def get_n_th_transition(self, indx_batch):
    #     # get the nth transition for each sample
    #     return [self.memory[cur_indx] for cur_indx in indx_batch+self.n_step]