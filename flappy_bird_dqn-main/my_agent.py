import numpy as np
import random
from collections import deque
import pygame
from pytorch_mlp import MLPRegression
import argparse
from console import FlappyBirdEnv

STUDENT_ID = 'a1897914'
DEGREE = 'UG'


class MyAgent:
    def __init__(self, show_screen=False, load_model_path=None, mode=None):
        self.show_screen = show_screen
        if mode is None:
            self.mode = 'train'
        else:
            self.mode = mode

        
        self.epsilon = 0.9
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.05
        self.n = 32 
        self.discount_factor = 0.95

        self.storage = deque(maxlen=2000)

        input_dim = 5  
        output_dim = 2  
        lr = 1e-3

        self.network = MLPRegression(input_dim=input_dim, output_dim=output_dim, learning_rate=lr)
        self.network2 = MLPRegression(input_dim=input_dim, output_dim=output_dim, learning_rate=lr)
        MyAgent.update_network_model(net_to_update=self.network2, net_as_source=self.network)

        if load_model_path:
            self.load_model(load_model_path)

        self.prev_state_vector = None
        self.prev_state = None
        self.prev_action = None

    def BUILD_STATE(self, state):
        bird_y = state['bird_y'] / state['screen_height']
        bird_velocity = (state['bird_velocity'] + 10) / 20
        pipe = state['pipes'][0] if state['pipes'] else {'x': state['screen_width'], 'top': 0, 'bottom': state['screen_height']}
        pipe_x_dist = (pipe['x'] - state['bird_x']) / state['screen_width']
        pipe_top = pipe['top'] / state['screen_height']
        pipe_bottom = pipe['bottom'] / state['screen_height']
        return np.array([bird_y, bird_velocity, pipe_x_dist, pipe_top, pipe_bottom])

    def REWARD(self, prev_state, next_state):
        if next_state['done_type'] in ['hit_pipe', 'offscreen']:
            return -100
        elif next_state['score'] > prev_state['score']:
            return 50
        else:
            return 1

    def choose_action(self, state: dict, action_table: dict) -> int:
        state_vector = self.BUILD_STATE(state)
        self.prev_state_vector = state_vector
        self.prev_state = state

        if self.mode == 'train' and np.random.rand() < self.epsilon:
            action = random.choice(list(action_table.values())[:2])  # jump or do_nothing
        else:
            q_values = self.network.predict(state_vector.reshape(1, -1))
            action = np.argmax(q_values)

        self.prev_action = action
        return action

    def receive_after_action_observation(self, state: dict, action_table: dict) -> None:
        if self.mode == 'eval':
            return

        next_state_vector = self.BUILD_STATE(state)
        reward = self.REWARD(self.prev_state, state)
        done = state['done']

        q_next = self.network2.predict(next_state_vector.reshape(1, -1))
        q_next_max = np.max(q_next)
        target_q = reward if done else reward + self.discount_factor * q_next_max

        self.storage.append((self.prev_state_vector, self.prev_action, target_q))

        if len(self.storage) >= self.n:
            self.replay_experience()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def replay_experience(self):
        batch = random.sample(self.storage, self.n)
        X, Y, W = [], [], []
        for state_vec, action, target_q in batch:
            q_values = self.network.predict(state_vec.reshape(1, -1)).flatten()
            q_values[action] = target_q
            X.append(state_vec)
            Y.append(q_values)
            W.append([1, 1])
        self.network.fit_step(np.array(X), np.array(Y), np.array(W))

    def save_model(self, path: str = 'my_model.ckpt'):
        self.network.save_model(path=path)

    def load_model(self, path: str = 'my_model.ckpt'):
        self.network.load_model(path=path)

    @staticmethod
    def update_network_model(net_to_update: MLPRegression, net_as_source: MLPRegression):
        net_to_update.load_state_dict(net_as_source.state_dict())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--level', type=int, default=1)
    args = parser.parse_args()

    env = FlappyBirdEnv(config_file_path='config.yml', show_screen=True, level=args.level, game_length=10)
    agent = MyAgent(show_screen=True)
    episodes = 10000
    for episode in range(episodes):
        env.play(player=agent)
        print(env.score)
        print(env.mileage)
        agent.save_model(path='my_model.ckpt')

        if episode % 50 == 0: 
            MyAgent.update_network_model(agent.network2, agent.network)
        if episode % 100 == 0: 
            agent.storage.clear()

    env2 = FlappyBirdEnv(config_file_path='config.yml', show_screen=False, level=args.level)
    agent2 = MyAgent(show_screen=False, load_model_path='my_model.ckpt', mode='eval')

    episodes = 10
    scores = []
    for episode in range(episodes):
        env2.play(player=agent2)
        scores.append(env2.score)

    print(np.max(scores))
    print(np.mean(scores))
