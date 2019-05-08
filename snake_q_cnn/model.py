import torch
from torch import nn
from torch.distributions import Categorical
from utils import *

class Q(nn.Module):
    '''state, action -> reward'''
    def __init__(self, num_actions, num_channels=4, hidden_dims=256):
        super(Q, self).__init__()
        world_size = 10
        self.num_actions = num_actions

        self.conv = nn.Sequential(
            # input size (1, 10, 10)
            nn.Conv2d(1, num_channels, 4, stride=2),
            nn.BatchNorm2d(num_channels),
            nn.LeakyReLU(.2, inplace=True),
            # state size (num_channels, 4, 4)
            nn.Conv2d(num_channels, num_channels*8, 3),
            nn.BatchNorm2d(num_channels*8),
            nn.LeakyReLU(.2, inplace=True),
            # state size (num_channels*8, 2, 2)
            nn.Conv2d(num_channels*8, num_channels*16, 2),
            # output size (num_channels*16, 1, 1)
        )

        self.fc = nn.Bilinear(num_channels*16, num_actions, hidden_dims)
        self.activation = nn.ReLU()
        self.out = nn.Sequential(
            nn.Dropout(.15),
            nn.Linear(hidden_dims, hidden_dims),
            self.activation,
            nn.Dropout(.15),
            nn.Linear(hidden_dims, 1),
        )
    
    def forward(self, state, action):
        '''
        state (batch, 1, 10, 10)
        action (batch,) gets one-hotted
        returns reward scalar tensor
        '''
        # multi argument bilinear in sequential call may cause bug
        conved = self.conv(state)
        conved_flat = conved.squeeze(3)
        conved_flat = conved_flat.squeeze(2)
        action = one_hot(action, self.num_actions) # now (batch, num_actions) one-hot
        fc = self.fc(conved_flat, action)
        activated = self.activation(fc)
        return self.out(activated)

    def choose_action(self, states):
        '''
        states (batch, 1, 10, 10)
        output (batch,)
        '''
        batch_size = states.shape[0]
        best_actions = torch.zeros((batch_size)).long().to(device)
        best_values = torch.zeros((batch_size)).float().to(device)
        for i, state in enumerate(states):
            state = state.unsqueeze(0)
            best_action = 0
            best_value = float('-inf')
            for action in range(self.num_actions):
                value = self.forward(state, [action])
                if value > best_value:
                    best_value = value
                    best_action = action
            best_actions[i] = best_action
            best_values[i] = best_value
        return best_actions, best_value

if __name__ == '__main__':
    # example usage
    q = Q(state_size=3, num_actions=2).to(device)
    state = torch.randn((1, 3)).to(device)
    action = [0]
    value = q(state, action)
    best_action = q.choose_action(state)
    best_value = q(state, q.choose_action(state))
    print(state, action, value, best_action, best_value, sep='\n')

