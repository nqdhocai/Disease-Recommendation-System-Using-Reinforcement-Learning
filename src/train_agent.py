from environment import Env
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tensorboardX import SummaryWriter
from collections import deque
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

class DQN(nn.Module):
    def __init__(self, input_size, hidden_size=100, action_size=2):
        super(DQN, self).__init__()

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, action_size)

    def forward(self, state):
        Q = self.fc1(state)
        Q = self.fc2(state)

        return Q

    def select_action(self, state):
        with torch.no_grad():
            Q = self.forward(state)
            action_index = torch.argmax(Q, dim=1)
        return action_index.item()

class Memory(object):
    def __init__(self, memory_size: int) -> None:
        self.memory_size = memory_size
        self.buffer = deque(maxlen=self.memory_size)

    def add(self, experience) -> None:
        self.buffer.append(experience)

    def size(self):
        return len(self.buffer)

    def sample(self, batch_size: int, continuous: bool = True):
        if batch_size > len(self.buffer):
            batch_size = len(self.buffer)
        if continuous:
            rand = random.randint(0, len(self.buffer) - batch_size)
            return [self.buffer[i] for i in range(rand, rand + batch_size)]
        else:
            indexes = np.random.choice(np.arange(len(self.buffer)), size=batch_size, replace=False)
            return [self.buffer[i] for i in indexes]

    def clear(self):
        self.buffer.clear()

if __name__ == '__main__':

    env = Env()
    input_size = env.state_embed_size
    onlineQNetwork = DQN(input_size).to(device)
    targetQNetwork = DQN(input_size).to(device)
    targetQNetwork.load_state_dict(onlineQNetwork.state_dict())

    optimizer = torch.optim.Adam(onlineQNetwork.parameters(), lr=1e-4)

    GAMMA = 0.99
    EPSILON_DECAY = 0.999
    INITIAL_EPSILON = 1
    FINAL_EPSILON = 0.001
    REPLAY_MEMORY = 50000
    BATCH = 512
    UPDATE_STEPS = 4
    EPOCHS = 50000

    memory_replay = Memory(REPLAY_MEMORY)
    epsilon = INITIAL_EPSILON
    learn_steps = 0
    writer = SummaryWriter('logs/dqn')
    begin_learn = False

    # for epoch in count():
    for epoch in range(EPOCHS):
        state = env.reset()

        episode_reward = 0
        for time_steps in range(200):
            p = random.random()
            if p < epsilon:
                action = random.randint(0, 1)
            else:
                tensor_state = state.unsqueeze(0).to(device)
                #             print(tensor_state)
                action = onlineQNetwork.select_action(tensor_state)
            next_state, reward, done = env.step(action)

            print("Reward: {}".format(reward))

            episode_reward += reward
            memory_replay.add((state, next_state, action, reward, done))
            if memory_replay.size() >= REPLAY_MEMORY:
                if begin_learn is False:
                    print('learn begin!')
                    begin_learn = True
                learn_steps += 1
                if learn_steps % UPDATE_STEPS == 0:
                    targetQNetwork.load_state_dict(onlineQNetwork.state_dict())
                batch = memory_replay.sample(BATCH, True)
                batch_state, batch_next_state, batch_action, batch_reward, batch_done = zip(*batch)

                batch_state = torch.stack(batch_state).to(device)
                batch_next_state = torch.stack(batch_next_state).to(device)
                batch_action = torch.FloatTensor(batch_action).unsqueeze(1).to(device)
                batch_reward = torch.FloatTensor(batch_reward).unsqueeze(1).to(device)
                batch_done = torch.FloatTensor(batch_done).unsqueeze(1).to(device)

                with torch.no_grad():
                    targetQ_next = targetQNetwork(batch_next_state)
                    y = batch_reward + (1 - batch_done) * GAMMA * torch.max(targetQ_next, dim=1, keepdim=True)[0]

                loss = F.mse_loss(onlineQNetwork(batch_state).gather(1, batch_action.long()), y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                writer.add_scalar('loss', loss.item(), global_step=learn_steps)

                if epsilon > FINAL_EPSILON:
                    epsilon *= EPSILON_DECAY

            if done:
                print(f'Epoch: {epoch} | Start symptom choice: {env.symptom} | Predict disease: {env._pred_diseases()}')
                break
            state = next_state

        writer.add_scalar('episode reward', episode_reward, global_step=epoch)
        if epoch % 100 == 0:
            torch.save(onlineQNetwork.state_dict(), 'dqn-policy.pt')
        if epoch % 1000 == 0:
            print('Ep {}\tMoving average score: {:.2f}\t'.format(epoch, episode_reward))