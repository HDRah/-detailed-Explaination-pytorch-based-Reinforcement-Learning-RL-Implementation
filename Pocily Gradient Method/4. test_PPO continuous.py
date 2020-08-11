import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
import matplotlib.pyplot as plt
import copy
import collections
import random
from torch.distributions import Categorical
from torch.distributions import Normal
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from torch import optim
# it is actually Dueling DDQN
# hyper-parameters

GAMMA = 0.9

op = np.finfo(np.float32).eps.item()


# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')
env = gym.make("Pendulum-v0")
env = env.unwrapped
#由于是连续性的动作，我们需要知道这个动作最大是多少
MAX_ACTION = float(env.action_space.high[0])
NUM_STATES = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

Buffer_SZ = 1000
UPDATE_PERIOD = 1

BATCH_SIZE = 32

update_times = 10
max_grad_norm = 0.5
epsilon = 0.2

Transition = collections.namedtuple('Transition', ['state', 'action',  'log_action_probs', 'reward', 'next_state']) #这是个好东西

def setup_seed(seed):
    env.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    # cudnn.benchmark = False
    # cudnn.enabled = False

setup_seed(1111)

class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(NUM_STATES, 100)
        self.fcmu= nn.Linear(100, action_dim)
        self.fcstd = nn.Linear(100, action_dim)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        mu = torch.tanh(self.fcmu(x)) * 2.0
        std = torch.sigmoid(self.fcstd(x))

        return (mu, std)

class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(NUM_STATES, 100)
        self.fc2 = nn.Linear(100, 1)

    def forward(self, state):
        out = F.relu(self.fc1(state))
        Value = self.fc2(out)

        return Value

class PPO():
    def __init__(self):
        self.actor = Actor().to(device)
        self.critic = Critic().to(device)

        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr = 1e-4)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr = 3e-4)

        self.Buffer = []
        self.counter = 0

    def choose_action(self, state):
        state = torch.Tensor(state).unsqueeze(0).to(device)

        with torch.no_grad():
            (mu, sigma) = self.actor(state)
        dist = Normal(mu, sigma)
        action = dist.sample()
        action_log_prob = dist.log_prob(action)
        action = action.clamp(-2.0, 2.0)
        return action.item(), action_log_prob.item()

    def store_transition(self, transition):
        self.Buffer.append(transition)
        self.counter += 1
        return self.counter % Buffer_SZ == 0

    def update(self):
        self.actor.train()
        self.critic.train()

        rewards = [t.reward for t in self.Buffer]
        old_log_action_probs = torch.Tensor([t.log_action_probs for t in self.Buffer]).to(device).view(-1, 1)
        states = torch.Tensor([t.state for t in self.Buffer]).to(device)
        next_states = torch.Tensor([t.next_state for t in self.Buffer]).to(device)
        old_actions = torch.Tensor([t.action for t in self.Buffer]).to(device).view(-1, 1)

        G = 0
        G_list = []
        for r in rewards[::-1]:
            G = GAMMA * G + r
            G_list.insert(0, G)
        G_list = torch.Tensor(G_list).to(device)
        G_list = (G_list - G_list.mean())/(G_list.std() + 1e-5)
        target_v = G_list.view(-1, 1).detach()

        Advan = (target_v - self.critic(states)).detach()

        for i in range(update_times):
            for index in BatchSampler(SubsetRandomSampler(range(len(self.Buffer))), BATCH_SIZE, False):

                #PPO核心
                (mu, sigma) = self.actor(states[index])
                dist = Normal(mu, sigma)
                new_log_action_prob = dist.log_prob(old_actions[index])

                entropy = dist.entropy()

                ratio = torch.exp(new_log_action_prob - old_log_action_probs[index])
                surr1 = ratio * Advan[index]
                surr2 = torch.clamp(ratio, 1-epsilon, 1+epsilon) * Advan[index]

                actor_loss = -torch.min(surr1, surr2).mean()# - 0.002*torch.mean(entropy)

                self.optimizer_actor.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), max_grad_norm)
                self.optimizer_actor.step()

                #更新critic网络
                critic_loss = F.smooth_l1_loss(self.critic(states[index]), target_v[index])
                self.optimizer_critic.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), max_grad_norm)
                self.optimizer_critic.step()

        del self.Buffer[:]

def main():
    PPO_Agent = PPO()

    episodes = 2000
    max_time = 200
    plt.ion()
    fig, ax = plt.subplots()
    rewards_list = []
    running_reward = -1000
    running_reward_list = []


    for i_episode in range(episodes):
        state = env.reset()
        ep_rewards = 0

        for t in range(max_time):  # Don't infinite loop while learning
            # env.render()
            action, action_log_prob = PPO_Agent.choose_action(state)
            next_state, reward, done, _ = env.step([action])

            ep_rewards += reward

            trans = Transition(state, action, action_log_prob, (reward + 8) / 8, next_state)
            if PPO_Agent.store_transition(trans):
                PPO_Agent.update()
            state = next_state

        running_reward = running_reward * 0.9 + ep_rewards * 0.1
        rewards_list.append(ep_rewards)
        running_reward_list.append(running_reward)

        if running_reward > -200:
            break
        ax.plot(rewards_list, 'g-', label='total_loss')
        plt.pause(0.001)

        if i_episode % 10 == 0:
            print("episode: {}, running reward is: {}, episode_rewards: {}".\
                  format(i_episode, round(running_reward, 3), round(ep_rewards)))

    plt.figure()
    plt.plot(running_reward_list, 'r')
    plt.legend(['rewards'])

if __name__ == '__main__':
    main()
b