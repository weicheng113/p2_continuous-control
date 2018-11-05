import torch.nn.functional as F
import torch.nn as nn
import torch


def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, fc1_units, fc2_units, seed):
        super(Actor, self).__init__()

        torch.manual_seed(seed)

        self.fc1 = nn.Linear(state_dim, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.output = nn.Linear(in_features=fc2_units, out_features=action_dim)

        self.bn1 = nn.BatchNorm1d(num_features=state_dim)
        # self.bn2 = nn.BatchNorm1d(num_features=fc1_units)
        # self.bn3 = nn.BatchNorm1d(num_features=fc2_units)

        init_weights(self.fc1)
        init_weights(self.fc2)
        init_weights(self.output)

    def forward(self, state):
        x = self.bn1(state)
        x = F.relu(self.fc1(x))

        # x = self.bn2(x)
        x = F.relu(self.fc2(x))

        # x = self.bn3(x)
        x = self.output(x)
        return F.tanh(x)


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, fc1_units, fc2_units, seed):
        super(Critic, self).__init__()

        torch.manual_seed(seed)

        self.fc1 = nn.Linear(in_features=state_dim, out_features=fc1_units)
        self.fc2 = nn.Linear(in_features=fc1_units+action_dim, out_features=fc2_units)
        self.output = nn.Linear(in_features=fc2_units, out_features=1)
        self.bn1 = nn.BatchNorm1d(num_features=state_dim)

        init_weights(self.fc1)
        init_weights(self.fc2)
        init_weights(self.output)

    def forward(self, state, action):
        x = self.bn1(state)
        x = F.relu(self.fc1(x))

        x = torch.cat((x, action), dim=1)
        x = F.relu(self.fc2(x))

        return self.output(x)

#
# class Critic(nn.Module):
#     def __init__(self, state_dim, action_dim, hidden_units, seed):
#         super(Critic, self).__init__()
#
#         torch.manual_seed(seed)
#
#         layer_sizes = zip((state_dim+action_dim, ) + hidden_units[:-1], hidden_units)
#         self.fc_layers = nn.ModuleList([nn.Linear(in_features=i, out_features=j) for i, j in layer_sizes])
#
#         self.output = nn.Linear(in_features=hidden_units[-1], out_features=1)
#         self.bn1 = nn.BatchNorm1d(num_features=state_dim)
#
#         self.fc_layers.apply(init_weights)
#         init_weights(self.output)
#
#     def forward(self, state, action):
#         x = self.bn1(state)
#         x = torch.cat((x, action), dim=1)
#
#         for fc in self.fc_layers:
#             x = F.relu(fc(x))
#
#         return self.output(x)
