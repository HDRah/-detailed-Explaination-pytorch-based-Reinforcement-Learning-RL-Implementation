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
# it is actually Dueling DDQN
# hyper-parameters

LR = 0.01
GAMMA = 0.9

op = np.finfo(np.float32).eps.item()


# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')
env = gym.make("CartPole-v1")
env = env.unwrapped
NUM_ACTIONS = env.action_space.n
NUM_STATES = env.observation_space.shape[0]
ENV_A_SHAPE = 0 if isinstance(env.action_space.sample(), int) else env.action_space.sample.shape

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

setup_seed(1234)

class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(NUM_STATES, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, NUM_ACTIONS)

    def forward(self, state):
        x = torch.tanh(self.fc1(state))
        x = torch.tanh(self.fc2(x))
        x = self.fc3(x)

        policy = F.softmax(x, dim=1)
        return policy

class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(NUM_STATES, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, state):
        out = self.fc1(state)
        out = torch.tanh(out)
        out = torch.tanh(self.fc2(out))
        out = self.fc3(out)

        return out

class Agent(nn.Module):
    def __init__(self):
        super().__init__()
        self.actor = Actor().to(device)
        self.critic = Critic().to(device)

        self.optimizer_Actor = torch.optim.Adam(self.actor.parameters(), lr = 0.001)
        self.optimizer_Critic = torch.optim.Adam(self.critic.parameters(), lr = 0.01)
        self.loss = nn.SmoothL1Loss()

    def choose_action(self, state):
        state = torch.Tensor(state).unsqueeze(0)
        state = state.to(device)
        action_prob = self.actor(state)
        #action_prob [batch_sz, acion_dim]
        action_prob = Categorical(action_prob)
        action = action_prob.sample()
        # action [batch_sz]

        return action.cpu().item(), action_prob.log_prob(action)

    def learn(self, state, next_state, reward, log_prob, I):
        self.critic.train()
        self.actor.train()

        state = torch.Tensor(state).unsqueeze(0).to(device)
        next_state = torch.Tensor(next_state).unsqueeze(0).to(device)

        state = state.to(device)
        next_state = next_state.to(device)

        Vs = self.critic(state)
        Vs_next = self.critic(next_state)
        TD_error = reward + GAMMA*Vs_next - Vs

        #updata critic network

        loss_critic = self.loss(Vs, reward + GAMMA*Vs_next)
        self.optimizer_Critic.zero_grad()
        loss_critic.backward()
        self.optimizer_Critic.step()

        #update actor network
        TD_error = TD_error.detach()
        loss_actor = -log_prob * I * TD_error

        self.optimizer_Actor.zero_grad()
        loss_actor.backward()
        self.optimizer_Actor.step()


def reward_func(env, x, x_dot, theta, theta_dot):
    r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.5
    r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
    reward = r1 + r2
    return reward

def main():
    Actor_Critic = Agent()

    episodes = 500
    max_time = 200
    plt.ion()
    fig, ax = plt.subplots()
    rewards_list = []
    flag = []
    time_list = []

    for i_episode in range(episodes):
        state = env.reset()
        ep_rewards = 0
        I = 1

        for t in range(max_time):  # Don't infinite loop while learning
            # env.render()
            action, log_prob = Actor_Critic.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            x, x_dot, theta, theta_dot = state
            reward = reward_func(env, x, x_dot, theta, theta_dot)
            ep_rewards += reward

            Actor_Critic.learn(state, next_state, reward, log_prob, I)

            I *= GAMMA

            state = next_state
            if done:
                break

        rewards_list.append(ep_rewards)
        ax.plot(rewards_list, 'g-', label='total_loss')
        plt.pause(0.001)

        time_list.append(t+1)
        if (t+1) == max_time:
            flag.append(1)
        else:
            flag = []

        if len(flag) == 10:
            print('This Agent is well-trained')
            break

        if i_episode % 10 == 0:
            print("episode: {}, the episode reward is {}, it has played {} times".\
                  format(i_episode, round(ep_rewards, 3), t+1))
    plt.figure()
    plt.plot(time_list)
    plt.legend(['times it insists'])




if __name__ == '__main__':
    main()
