import pickle

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from matplotlib import style
from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import BoltzmannQPolicy
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

style.use('ggplot')

SIZE = 10
EPISODES = 50000
SHOW_EVERY = 3000

# 环境参数
epsilon = 0.6  # 随机概率
EPS_DECAY = 0.9998  # 随机概率变化率
DISCOUNT = 0.95
LEARNING_RATE = 0.1


class Cube:
    def __init__(self, size):
        self.size = size
        self.x = np.random.randint(0, self.size)
        self.y = np.random.randint(0, self.size)

    def __str__(self):  # 打印位置
        return f'{self.x},{self.y}'

    def __sub__(self, other):  # 计算距离
        return (self.x - other.x, self.y - other.y)

    def action(self, choice):
        if choice == 0:
            self.move(x=1, y=1)
        elif choice == 1:
            self.move(x=-1, y=1)
        elif choice == 2:
            self.move(x=-1, y=-1)
        elif choice == 3:
            self.move(x=-1, y=-1)
        elif choice == 4:
            self.move(x=0, y=1)
        elif choice == 5:
            self.move(x=0, y=-1)
        elif choice == 6:
            self.move(x=1, y=0)
        elif choice == 7:
            self.move(x=-1, y=0)
        elif choice == 8:
            self.move(x=0, y=0)

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def move(self, x=False, y=False):
        if not x:
            self.x += np.random.randint(-1, 2)
        else:
            self.x += x

        if not y:
            self.y += np.random.randint(-1, 2)
        else:
            self.y += y

        if self.x < 0:
            self.x = 0
        elif self.x >= self.size:
            self.x = self.size - 1

        if self.y < 0:
            self.y = 0
        elif self.y >= self.size:
            self.y = self.size - 1


class envCube():
    SIZE = 10
    # OBSERVATION_SPACE_VALUES = (SIZE, SIZE, 3)
    OBSERVATION_SPACE_VALUES = (4,)
    ACTION_SPACE_VALUES = 9
    RETURN_IMAGE = False

    FOOD_REWARD = 25
    ENEMY_PENALITY = 300
    MOVE_PENALITY = 1

    d = {1: (255, 0, 0),  # blue
         2: (0, 255, 0),  # green
         3: (0, 0, 255)}  # red
    PLAYER_N = 1
    FOOD_N = 2
    ENEMY_N = 3

    def reset(self):
        self.player = Cube(self.SIZE)
        self.food = Cube(self.SIZE)
        while self.food == self.player:
            self.food = Cube(self.SIZE)
        self.enemy = Cube(self.SIZE)
        while self.enemy == self.player or self.enemy == self.food:
            self.enemy = Cube(self.SIZE)

        if self.RETURN_IMAGE:
            observation = np.array(self.get_image())
        else:
            observation = (self.player - self.food) + (self.player - self.enemy)
        self.episode_step = 0
        return observation

    def step(self, action):
        self.episode_step += 1
        self.player.action(action)
        self.food.move()
        self.enemy.move()
        if self.RETURN_IMAGE:
            new_obs = np.array(self.get_image())
        else:
            new_obs = (self.player - self.food) + (self.player - self.enemy)

        if self.player == self.food:
            reward = self.FOOD_REWARD
        elif self.player == self.enemy:
            reward = -self.ENEMY_PENALITY
        else:
            reward = -self.MOVE_PENALITY

        done = False
        if self.player == self.food or self.player == self.enemy or self.episode_step >= 200:
            done = True
        return new_obs, reward, done

    def render(self, t=1):
        img = self.get_image()
        img = img.resize((800, 800))
        cv2.imshow('Predator', np.array(img))
        cv2.waitKey(t)

    def get_qtable(self, qtable_name=None):
        if qtable_name is None:
            q_table = {}
            for x1 in range(-self.SIZE + 1, self.SIZE):
                for y1 in range(-self.SIZE + 1, self.SIZE):
                    for x2 in range(-self.SIZE + 1, self.SIZE):
                        for y2 in range(-self.SIZE + 1, self.SIZE):
                            q_table[(x1, y1, x2, y2)] = [np.random.uniform(-5, 0) for i in
                                                         range(self.ACTION_SPACE_VALUES)]
        else:
            with open(qtable_name, 'rb') as f:
                q_table = pickle.load(f)
                print("取得表格")
        return q_table

    def get_image(self):
        env = np.zeros((self.SIZE, self.SIZE, 3), dtype=np.uint8)
        env[self.player.x][self.player.y] = self.d[self.PLAYER_N]
        env[self.food.x][self.food.y] = self.d[self.FOOD_N]
        env[self.enemy.x][self.enemy.y] = self.d[self.ENEMY_N]

        img = Image.fromarray(env, 'RGB')
        return img

    def __init__(self):
        pass


class draw_picture():
    def __init__(self, episode_rewards, SHOW_EVERY=3000):
        moving_avg = np.convolve(episode_rewards, np.ones((SHOW_EVERY,)) / SHOW_EVERY, mode='valid')  # 每三个数求平均 卷积
        plt.plot([i for i in range(len(moving_avg))], moving_avg)
        plt.xlabel('episode #')
        plt.ylabel(f'mean {SHOW_EVERY} reward')
        plt.show()


def train(epsilon=0.5):
    env = envCube()
    model = build_model(env.OBSERVATION_SPACE_VALUES, env.ACTION_SPACE_VALUES)
    dqn = bulid_agent(model, env.ACTION_SPACE_VALUES)
    print(dqn)


def build_model(status, nb_actions):
    model = Sequential()
    model.add(Flatten(input_shape=(1,) + status))  # 不是图像 直接进入平滑层 将输入（4个数值）拉平了，每次输入一条数据
    model.add(Dense(32, activation='relu'))  # 2个连接层，32个输出
    model.add(Dense(32, activation='relu'))
    model.add(Dense(nb_actions, activation='linear'))  # 输出层动作
    return model


def bulid_agent(model, nb_actions):
    memory = SequentialMemory(limit=50000, window_length=1)  # 每次拿一条
    policy = BoltzmannQPolicy()  # 选择策略
    dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=1000,
                   target_model_update=1e-2, policy=policy)
    dqn.compile(Adam(lr=1e-3), metrics=['mae'])
    return dqn


def test(q_table, episodes, show_enable):
    env = envCube()
    q_table = env.get_qtable(q_table)
    for episode in range(episodes):
        episode_rewards = []
        obs = env.reset()
        done = False
        episode_reward = 0
        avg_reward = 0
        episode_step = 0
        while not done:
            action = np.argmax(q_table[obs])
            new_obs, reward, done = env.step(action)
            if show_enable == True:
                env.render(1)
            obs = new_obs
            episode_reward += reward
            episode_step += 1
            print(f'episode #{episode_step}, episode_reward:{episode_reward}')
    avg_reward += episode_reward
    avg_reward /= episodes
    print(f'All_episode #{episode_step}, ALL_episode_reward:{episode_reward},AVG_reward:{avg_reward}')


if __name__ == '__main__':
    train()
    # test("q_table_1639638457.pickle",10,True)
