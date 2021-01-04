import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.optim.rmsprop import RMSprop
from .ReplayMemory import ReplayMemory
import random
import numpy as np

def truncated_normal_(tensor,mean=0, std=0.09):
    with torch.no_grad():
        size = tensor.shape
        tmp = tensor.new_empty(size+(4,)).normal_()
        valid = (tmp < 2) & (tmp > -2)
        ind = valid.max(-1, keepdim=True)[1]
        tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
        tensor.data.mul_(std).add_(mean)
        return tensor

def weigth_init(m):
    if isinstance(m, nn.Linear):
        m.weight = truncated_normal_(m.weight, 0, 0.1)
        m.bias = truncated_normal_(m.bias, 0, 0.1)


class DNN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(DNN, self).__init__()
        self.fc1 = nn.Linear(in_features=input_shape, out_features=500)
        self.bn1 = nn.BatchNorm1d(num_features=500)
        self.fc2 = nn.Linear(in_features=500, out_features=250)
        self.bn2 = nn.BatchNorm1d(num_features=250)
        self.fc3 = nn.Linear(in_features=250, out_features=120)
        self.bn3 = nn.BatchNorm1d(num_features=120)
        self.fc4 = nn.Linear(in_features=120, out_features=num_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.bn1(x)
        x = F.relu(self.fc2(x))
        x = self.bn2(x)
        x = F.relu(self.fc3(x))
        x = self.bn3(x)
        x = F.relu(self.fc4(x))
        return x

class DQNAgent():

    def __init__(self, config, n_Feature, n_Action):
        self.device = config["device"]
        self.lr = config["lr"]
        self.target_net_update_freq = config["target_net_update_freq"]
        self.experience_replay_size = config["experience_replay_size"]
        self.batch_size = config["batch_size"]

        self.discount = config["discount"]


        self.num_feature = n_Feature
        self.num_action = n_Action

        # initialize the predict net and target net in the agent
        self.predict_net = DNN(input_shape=n_Feature, num_actions=n_Action)
        self.target_net = DNN(input_shape=n_Feature, num_actions=n_Action)
        self.predict_net = self.predict_net.to(self.device)
        self.target_net = self.target_net.to(self.device)
        self.predict_net.apply(weigth_init)
        self.target_net.apply(weigth_init)
        self.target_net.load_state_dict(self.predict_net.state_dict())
        self.optimizer = RMSprop(self.predict_net.parameters(), lr=self.lr, momentum=0.95, eps=0.01)
        # self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        # initialize the experience replay buffer
        self.memory = ReplayMemory(entry_size=n_Feature, memory_size=self.experience_replay_size, batch_size=self.batch_size)


    def prepare_minibatch(self):
        batch_state, batch_next_state, batch_action, batch_reward = self.memory.sample()
        batch_state = torch.tensor(batch_state, device=self.device, dtype=torch.float)
        batch_action = torch.tensor(batch_action, device=self.device, dtype=torch.long).view(-1, 1)
        batch_reward = torch.tensor(batch_reward, device=self.device, dtype=torch.float)
        batch_next_state = torch.tensor(batch_next_state, device=self.device, dtype=torch.float)
        return batch_state, batch_action, batch_reward, batch_next_state


    def update_dqn(self, time_step):
        batch_state, batch_action, batch_reward, batch_next_state = self.prepare_minibatch()
        current_state_values = self.predict_net(batch_state).gather(1, batch_action).squeeze()
        next_state_values = self.target_net(batch_next_state).max(1)[0].detach()
        expected_state_action_values = (next_state_values * self.discount) + batch_reward
        # compute temporal difference as loss
        loss = (expected_state_action_values - current_state_values)**2
        loss = loss.mean()
        # optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # update target network
        if time_step % self.target_net_update_freq == self.target_net_update_freq - 1:
            self.target_net.load_state_dict(self.predict_net.state_dict())
        return loss.item()


    def update_double_dqn(self, time_step):
        batch_state, batch_action, batch_reward, batch_next_state = self.prepare_minibatch()
        current_state_values = self.predict_net(batch_state).gather(1, batch_action).squeeze()
        pred_action = self.predict_net(batch_next_state).argmax(1).unsqueeze(1)
        next_state_values = self.target_net(batch_next_state).gather(1, pred_action).squeeze()
        expected_state_action_values = (next_state_values * self.discount) + batch_reward
        # compute temporal difference as loss
        loss = (expected_state_action_values - current_state_values) ** 2
        loss = loss.mean()
        # optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # update target network
        if time_step % self.target_net_update_freq == self.target_net_update_freq - 1:
            self.target_net.load_state_dict(self.predict_net.state_dict())
        return loss.item()


    def get_action(self, state, epsilon, is_static_policy=False):
        with torch.no_grad():
            rand = random.random()
            if rand < epsilon or not is_static_policy:
                return np.random.randint(0, self.num_action)
            else:
                X = torch.tensor([state], device=self.device, dtype=torch.float)
                a = self.predict_net.forward(X).squeeze().argmax().item()
                return a