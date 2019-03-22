# -*- coding: utf-8 -*-
"""
An implementation of the policyValueNet in PyTorch
Tested in PyTorch 0.2.0 and 0.3.0
@author: Junxiao Song
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


def set_learning_rate(optimizer, lr):
    """Sets the learning rate to the given value"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class Net(nn.Module):
    """policy-value network module"""
  #nn.BatchNorm1d(n_hidden_1)
    def __init__(self, in_dim, n_hidden_1, n_hidden_2, out_dim):
        super(Net, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(in_dim, n_hidden_1),
                                   nn.Sigmoid())
        self.layer2 = nn.Sequential(nn.Linear(n_hidden_1, n_hidden_2),
                                    nn.Sigmoid())
        self.layer3 = nn.Sequential(nn.Linear(n_hidden_2, out_dim),
                                    nn.ReLU())

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


class PolicyValueNet():
    """policy-value network """
    def __init__(self, in_dim, n_hidden_1, n_hidden_2, out_dim,
                 model_file=None, use_gpu=True):
        self.use_gpu = torch.cuda.is_available()
        self.in_dim = in_dim
        self.n_hidden_1 = n_hidden_1
        self.n_hidden_2 = n_hidden_2
        self.out_dim = out_dim

        self.l2_const = 1e-4  # coef of l2 penalty
        # the policy value net module
        if self.use_gpu:
            self.policy_value_net = Net(in_dim, n_hidden_1,n_hidden_2,out_dim).cuda()
        else:
            self.policy_value_net = Net(in_dim, n_hidden_1,n_hidden_2,out_dim)
        self.optimizer = optim.Adam(self.policy_value_net.parameters(),
                                    weight_decay=self.l2_const)

        if model_file:
            net_params = torch.load(model_file)
            self.policy_value_net.load_state_dict(net_params)

    def policy_value(self, state_batch):
        """
        input: a batch of states
        output: a batch of action probabilities and state values
        """
        if self.use_gpu:
            state_batch = Variable(torch.FloatTensor(state_batch).cuda())
            log_act_probs = self.policy_value_net(state_batch)
            act_probs = log_act_probs.data.cpu().numpy().flatten()
            return act_probs
        else:
            state_batch = Variable(torch.FloatTensor(state_batch))
            log_act_probs = self.policy_value_net(state_batch)
            act_probs = log_act_probs.data.numpy().flatten()
            return act_probs

    def train_step(self, state_batch, mcts_probs, lr):
        """perform a training step"""
        # wrap in Variable
        if self.use_gpu:
            state_batch = Variable(torch.FloatTensor(state_batch).cuda())
            mcts_probs = Variable(torch.FloatTensor(mcts_probs).cuda())
        else:
            state_batch = Variable(torch.FloatTensor(state_batch))
            mcts_probs = Variable(torch.FloatTensor(mcts_probs))

        # zero the parameter gradients
        self.optimizer.zero_grad()
        # set learning rate
        set_learning_rate(self.optimizer, lr)

        # forward
        log_act_probs = self.policy_value_net(state_batch)
        # define the loss = (z - v)^2 - pi^T * log(p) + c||theta||^2
        # Note: the L2 penalty is incorporated in optimizer
        policy_loss = torch.sum(torch.abs(mcts_probs-log_act_probs.squeeze()))
        loss = policy_loss
        # backward and optimize
        loss.backward()
        self.optimizer.step()
        # calc policy entropy, for monitoring only

        # return loss.data[0], entropy.data[0]
        # for pytorch version >= 0.5 please use the following line instead.
        return loss.item(), 0

    def get_policy_param(self):
        net_params = self.policy_value_net.state_dict()
        return net_params

    def save_model(self, model_file):
        """ save model params to file """
        net_params = self.get_policy_param()  # get model params
        torch.save(net_params, model_file)