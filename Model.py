import torch
import torch as th
from torch import nn

NUMBER = 4

class ActorNetwork(nn.Module):
    def __init__(self, state_dim, action_dim=26):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, action_dim)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)
        nn.init.zeros_(self.fc3.bias)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        probs = self.softmax(x)
        return probs

    # def __init__(self, state_dim, output_size=27, output_act=nn.functional.softmax, init_w=3e-3):  # 改为 27
    #     super(ActorNetwork, self).__init__()
    #     self.fc1 = nn.Linear(state_dim, 64)
    #     self.fc2 = nn.Linear(64, 128)
    #     self.fc3 = nn.Linear(128, output_size)
    #     self.fc3.weight.data.uniform_(-init_w, init_w)
    #     self.fc3.bias.data.uniform_(-init_w, init_w)
    #     self.output_act = output_act
    #
    # def __call__(self, state):
    #     out = nn.functional.relu(self.fc1(state))
    #     out = nn.functional.relu(self.fc2(out))
    #     out = self.output_act(self.fc3(out), dim=-1)
    #     return out

class CriticNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, output_size=1, init_w=3e-3):
        super(CriticNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, output_size)
        self.fc3.weight.data.uniform_(-init_w, init_w)
        self.fc3.bias.data.uniform_(-init_w, init_w)

    def __call__(self, state, action):
        assert state.size(0) == action.size(0), f"批次大小不匹配：state {state.size(0)}, action {action.size(0)}"
        out = th.cat([state, action], 1)
        out = nn.functional.relu(self.fc1(out))
        out = nn.functional.relu(self.fc2(out))
        out = self.fc3(out)
        return out