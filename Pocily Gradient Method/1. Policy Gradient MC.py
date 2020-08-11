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
GAMMA = 0.99

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

class softmaxPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(NUM_STATES, 128)
        self.fc2 = nn.Linear(128, NUM_ACTIONS)

    def forward(self, state):
        x = self.fc1(state)
        x = F.relu(x)
        x = self.fc2(x)

        policy = F.softmax(x, dim=1)
        return policy

class Agent(nn.Module):
    def __init__(self):
        super().__init__()
        self.policy = softmaxPolicy().to(device)

        self.rewards = []
        self.log_policy = []
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr = LR)

    def choose_action(self, state):
        state = torch.Tensor(state).unsqueeze(0)
        state = state.to(device)
        action_prob = self.policy(state)
        #action_prob [batch_sz, acion_dim]
        action_prob = Categorical(action_prob)
        action = action_prob.sample()
        # action [batch_sz]
        self.log_policy.append(action_prob.log_prob(action))

        return action.cpu().item()

    def learn(self):
        self.policy.train()
        G_list = []
        G = 0
        episode_loss = []
        for r in reversed(self.rewards):
            G = r + G*GAMMA
            G_list.insert(0, G)
        G_list = torch.tensor(G_list).to(device)
        G_list = (G_list - G_list.mean())/(G_list.std() + op)

        for log_prob, G in zip(self.log_policy, G_list):
            episode_loss.append(-log_prob * G)

        episode_loss = torch.cat(episode_loss).sum()
        self.optimizer.zero_grad()
        episode_loss.backward()
        self.optimizer.step()

        del self.rewards[:]
        del self.log_policy[:]

def reward_func(env, x, x_dot, theta, theta_dot):
    r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.5
    r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
    reward = r1 + r2
    return reward

def main():
    PG_MC = Agent()

    episodes = 500
    max_time = 500
    plt.ion()
    fig, ax = plt.subplots()
    rewards_list = []
    flag = []
    time_list = []

    for i_episode in range(episodes):
        state = env.reset()
        ep_rewards = 0

        for t in range(max_time):  # Don't infinite loop while learning
            # env.render()
            action = PG_MC.choose_action(state)
            state, reward, done, _ = env.step(action)
            x, x_dot, theta, theta_dot = state
            reward = reward_func(env, x, x_dot, theta, theta_dot)
            ep_rewards += reward

            PG_MC.rewards.append(reward)
            if done:
                break

        rewards_list.append(ep_rewards)
        ax.plot(rewards_list, 'g-', label='total_loss')
        plt.pause(0.001)
        PG_MC.learn()
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

