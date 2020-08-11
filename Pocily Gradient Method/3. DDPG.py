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
device = torch.device('cuda')
env = gym.make("Pendulum-v0")
env = env.unwrapped
#由于是连续性的动作，我们需要知道这个动作最大是多少
MAX_ACTION = float(env.action_space.high[0])
NUM_STATES = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
Experience_SZ = 50000
UPDATE_PERIOD = 1
tau = 0.005
BATCH_SIZE = 128
training_volumn = 50000
Learning_iteration = 10

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
        self.fc1 = nn.Linear(NUM_STATES, 400)
        self.fc2 = nn.Linear(400, 300)
        self.fc3 = nn.Linear(300, action_dim)

    def forward(self, state):

        out = F.relu(self.fc1(state))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        out = torch.tanh(out) * MAX_ACTION

        return out

class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(NUM_STATES + action_dim, 400)
        self.fc2 = nn.Linear(400, 300)
        self.fc3 = nn.Linear(300, 1)

    def forward(self, state):

        out = F.relu(self.fc1(state))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)

        return out

class Agent(nn.Module):
    def __init__(self, IF_TRAIN = True):
        super().__init__()
        self.actor = Actor().to(device)
        self.target_actor = Actor().to(device)
        self.target_actor.load_state_dict(self.actor.state_dict())  #很重要！！！

        self.critic = Critic().to(device)
        self.target_critic = Critic().to(device)
        self.target_critic.load_state_dict(self.critic.state_dict())

        self.Replay_Buffer = collections.deque(maxlen=Experience_SZ)

        self.optimizer_Actor = torch.optim.Adam(self.actor.parameters(), lr = 0.001)
        self.optimizer_Critic = torch.optim.Adam(self.critic.parameters(), lr = 0.001)
        self.loss = nn.MSELoss()

        self.update_times = 0
        self.if_train = IF_TRAIN

    def choose_action(self, state):

        state = torch.Tensor(state).unsqueeze(0).to(device)

        action = self.actor(state)
        #action [batch_sz, acion_dim]

        return action.cpu().data.numpy().flatten()

    def store_transition(self, state, reward, action, next_state, done):
        transition = np.hstack((state, action, [reward], next_state, [done]))
        self.Replay_Buffer.append(transition)

    def update_target_networks(self): #网络软更新的方法，通过parameters属性中的迭代更新实现
        for param, target_param in zip(self.critic.parameters(), self.target_critic.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.target_actor.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def learn(self):
        self.update_times += 1
        self.critic.train()
        self.actor.train()

        batch_sampled = torch.Tensor(random.sample(self.Replay_Buffer, BATCH_SIZE)).to(device)
        batch_states = batch_sampled[:, :NUM_STATES]
        batch_actions = batch_sampled[:, NUM_STATES:NUM_STATES + action_dim]
        batch_rewards = batch_sampled[:, NUM_STATES + action_dim:NUM_STATES + action_dim + 1]
        batch_nextstates = batch_sampled[:, NUM_STATES + action_dim + 1:2 * NUM_STATES + action_dim + 1]
        batch_done = batch_sampled[:, -1:]

        batch_nextactions = self.target_actor(batch_nextstates)

        target_Q = self.target_critic(torch.cat((batch_nextstates, batch_nextactions), dim=1))
        target_Q = batch_rewards + (1 - batch_done)*(GAMMA * target_Q)
        target_Q = target_Q.detach()

        Q = self.critic(torch.cat((batch_states, batch_actions), dim=1))

        critic_loss = self.loss(Q, target_Q)
        #更新critic网络
        self.optimizer_Critic.zero_grad()
        critic_loss.backward()
        self.optimizer_Critic.step()

        #更新actor网络
        actions = self.actor(batch_states) #这里太重要了！！！！！！！！！没有这行就没有梯度了
        actor_loss = -self.critic(torch.cat((batch_states, actions), dim=1)).mean()
        self.optimizer_Actor.zero_grad()
        actor_loss.backward()
        self.optimizer_Actor.step()

        self.update_target_networks()



def main():
    DDPG = Agent()
    print('collecting experience......')

    episodes = 500
    max_time = 2000
    plt.ion()
    fig, ax = plt.subplots()
    rewards_list = []
    time_list = []
    overall_rewards = 0
    flag = 0

    for i_episode in range(episodes):
        state = env.reset()
        ep_rewards = 0

        for t in range(max_time):  # Don't infinite loop while learning
            # env.render()
            action = DDPG.choose_action(state)
            action = (action + np.random.normal(0, 0.1, size=action_dim)).clip(\
                    env.action_space.low, env.action_space.high)
            next_state, reward, done, _ = env.step(np.float32(action)) #这样输入action
            ep_rewards += reward

            DDPG.store_transition(state, reward, action, next_state, done)

            state = next_state

            if done:
                break

        overall_rewards += ep_rewards

        if len(DDPG.Replay_Buffer) >= training_volumn - 1:
            if flag == 0:
                print('##############now start training##################')
                flag = 1
            for i_learn in range(Learning_iteration):
                DDPG.learn()

        rewards_list.append(ep_rewards)
        ax.plot(rewards_list, 'g-', label='total_loss')
        plt.pause(0.001)

        time_list.append(t+1)

        if i_episode % 10 == 0:
            print("episode: {}, the episode reward is: {}, overall rewards are: {}".\
                  format(i_episode, round(ep_rewards, 3), round(overall_rewards, 3)))
            overall_rewards = 0
    plt.figure()
    plt.plot(time_list)
    plt.legend(['times it insists'])




if __name__ == '__main__':
    main()