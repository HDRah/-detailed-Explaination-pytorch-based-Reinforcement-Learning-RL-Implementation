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
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from torch import optim
# it is actually Dueling DDQN
# hyper-parameters

GAMMA = 0.98

op = np.finfo(np.float32).eps.item()


# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cuda')
env = gym.make("CartPole-v1")
env = env.unwrapped

NUM_STATES = env.observation_space.shape[0]
action_dim = env.action_space.n
Experience_SZ = 50000
UPDATE_PERIOD = 1

BATCH_SIZE = 32

update_times = 10
max_grad_norm = 0.5
epsilon = 0.2

Transition = collections.namedtuple('Transition', ['state', 'action',  'action_probs', 'reward', 'next_state']) #这是个好东西

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

setup_seed(2333)

class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(NUM_STATES, 100)
        self.fc2 = nn.Linear(100, action_dim)

    def forward(self, state):
        out = F.relu(self.fc1(state))
        out = F.softmax(self.fc2(out), dim=1)

        return out

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

        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr = 1e-3)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr = 3e-3)

        self.Buffer = []

    def choose_action(self, state):
        state = torch.Tensor(state).unsqueeze(0).to(device)

        with torch.no_grad():
            action_prob = self.actor(state)
        c = Categorical(action_prob)
        action = c.sample()

        return action.item(), action_prob[:, action.item()].item()

    def store_transition(self, transition):
        self.Buffer.append(transition)

    def update(self):
        self.actor.train()
        self.critic.train()

        rewards = [t.reward for t in self.Buffer]
        old_action_probs = torch.Tensor([t.action_probs for t in self.Buffer]).to(device).view(-1, 1).detach()
        states = torch.Tensor([t.state for t in self.Buffer]).to(device).detach()
        old_actions = torch.Tensor([t.action for t in self.Buffer]).to(device).view(-1, 1).detach()
        old_actions = old_actions.long()


        G = 0
        G_list = []
        for r in rewards[::-1]:
            G = GAMMA*G + r
            G_list.insert(0, G)

        G_list = torch.Tensor(G_list).to(device)
        G_list = (G_list - G_list.mean())/(G_list.std() + 1e-5) #标准化收获
        G_list = G_list.view(-1, 1)
        for i in range(update_times):
            for index in BatchSampler(SubsetRandomSampler(range(len(self.Buffer))), BATCH_SIZE, False):
                V = self.critic(states[index])
                Advan = G_list[index] - V.detach()

                #PPO核心
                new_action_prob_all = self.actor(states[index])
                c = Categorical(new_action_prob_all)
                entropy = c.entropy()

                new_action_prob = new_action_prob_all.gather(1, old_actions[index])
                ratio = (new_action_prob / old_action_probs[index].detach())
                surr1 = ratio*Advan
                surr2 = torch.clamp(ratio, 1-epsilon, 1+epsilon) * Advan

                actor_loss = -torch.min(surr1, surr2).mean() - 0.002*torch.mean(entropy)

                self.optimizer_actor.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), max_grad_norm)
                self.optimizer_actor.step()

                #更新critic网络
                critic_loss = F.mse_loss(V, G_list[index])
                self.optimizer_critic.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), max_grad_norm)
                self.optimizer_critic.step()

        del self.Buffer[:]


def reward_func(env, x, x_dot, theta, theta_dot):
    r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.5
    r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
    reward = r1 + r2
    return reward


def main():
    PPO_Agent = PPO()

    episodes = 300
    max_time = 500
    plt.ion()
    fig, ax = plt.subplots()
    rewards_list = []
    time_list = []


    for i_episode in range(episodes):
        state = env.reset()
        ep_rewards = 0

        for t in range(max_time):  # Don't infinite loop while learning
            # env.render()
            action, action_prob = PPO_Agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            # x, x_dot, theta, theta_dot = state
            # reward = reward_func(env, x, x_dot, theta, theta_dot)
            ep_rewards += reward

            trans = Transition(state, action, action_prob, reward, next_state)
            PPO_Agent.store_transition(trans)
            state = next_state

            if done:
                if len(PPO_Agent.Buffer) >= BATCH_SIZE:
                    PPO_Agent.update()    #更新的位置及其重要
                break


        rewards_list.append(ep_rewards)
        ax.plot(rewards_list, 'g-', label='total_loss')
        plt.pause(0.001)

        time_list.append(t+1)

        if i_episode % 10 == 0:
            print("episode: {}, the episode reward is: {}".\
                  format(i_episode, round(ep_rewards, 3)))
            overall_rewards = 0
    plt.figure()
    plt.plot(time_list)
    plt.legend(['times it insists'])




if __name__ == '__main__':
    main()







