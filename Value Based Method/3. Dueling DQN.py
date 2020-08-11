import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
import matplotlib.pyplot as plt
import copy
import collections
import random
# it is actually Dueling DDQN
# hyper-parameters
BATCH_SIZE = 128
LR = 0.01
GAMMA = 0.9
MEMORY_CAPACITY = 4000
Q_NETWORK_ITERATION = 100


# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cuda')
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

class DDQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(NUM_STATES, 128)
        self.fc2 = nn.Linear(128, 128)

        self.V = nn.Linear(128, 1)
        self.A = nn.Linear(128, NUM_ACTIONS)

    def forward(self,x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        Value = self.V(x)
        Advantage = self.A(x)

        Q_value = Value + (Advantage - Advantage.mean(dim=1, keepdim=True))
        return Q_value


class Agent(nn.Module):
    """docstring for DQN"""

    def __init__(self):
        super().__init__()
        self.eval_net, self.target_net = DDQN(), DDQN()

        self.learn_step_counter = 0
        self.memory_counter = 0
        self.memory = collections.deque(maxlen=MEMORY_CAPACITY)
        # why the NUM_STATE*2 +2
        # When we store the memory, we put the state, action, reward and next_state in the memory
        # here reward and action is a number, state is a ndarray
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()

    def choose_action(self, state, epsilon):
        state = torch.unsqueeze(torch.FloatTensor(state), 0).to(device)  # get a 1D array
        if np.random.rand() > epsilon:  # greedy policy
            action_value = self.eval_net.forward(state)
            action = torch.max(action_value, 1)[1].cpu().data.numpy()
            action = action[0] if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)
        else:  # random policy
            action = np.random.randint(0, NUM_ACTIONS)
            action = action if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)
        return action

    def store_transition(self, state, action, reward, next_state):
        transition = np.hstack((state, [action, reward], next_state))
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory.append(transition)
        self.memory_counter += 1

    def learn(self):

        # update the parameters
        if self.learn_step_counter % Q_NETWORK_ITERATION == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        # sample batch from memory

        batch_memory = torch.Tensor(random.sample(self.memory, BATCH_SIZE)).to(device)

        batch_state = batch_memory[:, :NUM_STATES]
        batch_action = batch_memory[:, NUM_STATES:NUM_STATES + 1].long()
        batch_reward = batch_memory[:, NUM_STATES + 1:NUM_STATES + 2]
        batch_next_state = batch_memory[:, -NUM_STATES:]

        # q_eval
        q_eval = self.eval_net(batch_state).gather(1, batch_action)
        target_action = self.eval_net(batch_state).argmax(dim=1).unsqueeze(1).long()
        # 使用eval_net找到最大Q值对应的动作，再用此动作带入target_Qnet计算target_Q
        q_next = self.target_net(batch_next_state).gather(1, target_action).detach()
        # q_next:[batch_sz, 2 (action dim)]
        q_target = batch_reward + GAMMA * q_next
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


def reward_func(env, x, x_dot, theta, theta_dot):
    r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.5
    r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
    reward = r1 + r2
    return reward


def main():
    dqn = Agent().to(device)

    time_list = []
    episodes = 300
    max_time = 500
    print("Collecting Experience....")
    reward_list = []
    plt.ion()
    fig, ax = plt.subplots()
    for i in range(episodes):
        state = env.reset()
        epsilon = max(0.01, 0.1 - (0.1 - 0.01) * (i / 200))
        ep_reward = 0
        play_times = 0
        for t in range(max_time):
            env.render()
            action = dqn.choose_action(state, epsilon)
            next_state, reward, done, info = env.step(action)
            x, x_dot, theta, theta_dot = next_state
            reward = reward_func(env, x, x_dot, theta, theta_dot)

            dqn.store_transition(state, action, reward, next_state)
            ep_reward += reward
            play_times += 1

            if dqn.memory_counter >= 2000:
                dqn.learn()
                if done or t == max_time - 1:
                    print("episode: {} , epsilon: {} , the episode reward is {}, it has played {} times".\
                          format(i, round(epsilon, 3), round(ep_reward, 3), play_times))
            if done or t == max_time - 1:
                break
            state = next_state

        time_list.append(play_times)
        r = copy.copy(ep_reward)
        reward_list.append(r)
        # ax.set_xlim(0, 300)
        # ax.cla()
        ax.plot(reward_list, 'g-', label='total_loss')
        plt.pause(0.001)

    plt.figure()
    plt.plot(time_list)
    plt.legend(['times it insists'])


if __name__ == '__main__':
    main()