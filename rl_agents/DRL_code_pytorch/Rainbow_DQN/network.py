import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from functools import reduce
import numpy as np
from rl_agents.config import GPU_NAME

device = torch.device(GPU_NAME if torch.cuda.is_available() else "cpu")
print(device)
class Dueling_Net(nn.Module):
    def __init__(self, args):
        super(Dueling_Net, self).__init__()
        k1 = 2
        k2 = 3
        mid_channels = 3
        dimensions = np.array(args.state_dim)
        pool_dim = 2
        self.conv1 = nn.Conv2d(in_channels=1,out_channels=mid_channels, kernel_size=k1)
        self.pool = nn.MaxPool2d(pool_dim,pool_dim)
        self.conv2 = nn.Conv2d(in_channels=mid_channels,out_channels=1,kernel_size=k2)
        self.fc1 = nn.Linear(np.multiply(*(((dimensions-k1+1)/pool_dim).astype(int)-k2+1)), args.hidden_dim)
        self.fc2 = nn.Linear(args.hidden_dim, args.hidden_dim)
        if args.use_noisy:
            self.V = NoisyLinear(args.hidden_dim, 1)
            self.A = NoisyLinear(args.hidden_dim, args.action_dim)
        else:
            self.V = nn.Linear(args.hidden_dim, 1)
            self.A = nn.Linear(args.hidden_dim, args.action_dim)

    def forward(self, s):
        s = s.unsqueeze(1).to(device)
        s = self.pool(F.relu(self.conv1(s)))
        s = F.relu(self.conv2(s)).squeeze(0)
        s = torch.flatten(s,1)
        s = torch.relu(self.fc1(s))
        s = torch.relu(self.fc2(s))
        V = self.V(s)  # batch_size X 1
        A = self.A(s)  # batch_size X action_dim
        Q = V + (A - torch.mean(A, dim=-1, keepdim=True))  # Q(s,a)=V(s)+A(s,a)-mean(A(s,a))
        return Q


class Net(nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()
        k1 = 2
        k2 = 3
        mid_channels = 3
        dimensions = np.array(args.state_dim)
        pool_dim = 2
        self.conv1 = nn.Conv2d(in_channels=1,out_channels=mid_channels, kernel_size=k1)
        self.pool = nn.MaxPool2d(pool_dim,pool_dim)
        self.conv2 = nn.Conv2d(in_channels=mid_channels,out_channels=1,kernel_size=k2)
        self.fc1 = nn.Linear(np.multiply(*(((dimensions-k1+1)/pool_dim).astype(int)-k2+1)), args.hidden_dim)
        self.fc2 = nn.Linear(args.hidden_dim, args.hidden_dim)
        if args.use_noisy:
            self.fc3 = NoisyLinear(args.hidden_dim, args.action_dim)
        else:
            self.fc3 = nn.Linear(args.hidden_dim, args.action_dim)

    def forward(self, s):
        s = s.unsqueeze(1)
        s = self.pool(F.relu(self.conv1(s)))
        s = F.relu(self.conv2(s)).squeeze(0)
        s = torch.flatten(s, 1)
        s = torch.relu(self.fc1(s))
        s = torch.relu(self.fc2(s))
        Q = self.fc3(s)
        return Q


class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, sigma_init=0.5):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sigma_init = sigma_init

        self.weight_mu = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.FloatTensor(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.FloatTensor(out_features))
        self.bias_sigma = nn.Parameter(torch.FloatTensor(out_features))
        self.register_buffer('bias_epsilon', torch.FloatTensor(out_features))

        self.reset_parameters()
        self.reset_noise()

    def forward(self, x):
        if self.training:
            self.reset_noise()
            weight = self.weight_mu + self.weight_sigma.mul(self.weight_epsilon)  # mul是对应元素相乘
            bias = self.bias_mu + self.bias_sigma.mul(self.bias_epsilon)

        else:
            weight = self.weight_mu
            bias = self.bias_mu

        return F.linear(x, weight, bias)

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.bias_mu.data.uniform_(-mu_range, mu_range)

        self.weight_sigma.data.fill_(self.sigma_init / math.sqrt(self.in_features))
        self.bias_sigma.data.fill_(self.sigma_init / math.sqrt(self.out_features))  # 这里要除以out_features

    def reset_noise(self):
        epsilon_i = self.scale_noise(self.in_features)
        epsilon_j = self.scale_noise(self.out_features)
        self.weight_epsilon.copy_(torch.ger(epsilon_j, epsilon_i))
        self.bias_epsilon.copy_(epsilon_j)

    def scale_noise(self, size):
        x = torch.randn(size)  # torch.randn产生标准高斯分布
        x = x.sign().mul(x.abs().sqrt())
        return x
