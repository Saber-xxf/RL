import pickle
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from matplotlib import style

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
    OBSERVATION_SPACE_VALUES = (SIZE, SIZE, 3)
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

    def render(self,t=1):
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
    def __init__(self, episode_rewards,SHOW_EVERY=3000):
        moving_avg = np.convolve(episode_rewards, np.ones((SHOW_EVERY,)) / SHOW_EVERY, mode='valid')  # 每三个数求平均 卷积
        plt.plot([i for i in range(len(moving_avg))], moving_avg)
        plt.xlabel('episode #')
        plt.ylabel(f'mean {SHOW_EVERY} reward')
        plt.show()


def train(epsilon=0.5):
    env = envCube()
    q_table = env.get_qtable("q_table_1639638457.pickle")

    episode_rewards = []
    for episode in range(EPISODES):
        obs = env.reset()
        done = False
        if episode % SHOW_EVERY == 0:
            print(f'episode #{episode}, epsilon:{epsilon}')
            print(f'mean reward :{np.mean(episode_rewards[-SHOW_EVERY:])}')
            show = True
        else:
            show = False
        episode_reward = 0
        while not done:
            if np.random.random() > epsilon:
                action = np.argmax(q_table[obs])
            else:
                action = np.random.randint(0, env.ACTION_SPACE_VALUES)
            new_obs, reward, done = env.step(action)

            current_q = q_table[obs][action]
            max_future_q = np.max(q_table[new_obs])
            if reward == env.FOOD_REWARD:
                new_q = env.FOOD_REWARD
            else:
                new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
            q_table[obs][action] = new_q
            obs = new_obs
            if show:
                env.render(1)
            episode_reward += reward
        episode_rewards.append(episode_reward)
        epsilon *= EPS_DECAY

    draw_picture(episode_rewards)
    with open(f'q_table_{int(time.time())}.pickle', 'wb') as f:
        pickle.dump(q_table, f)


def test():
    env = envCube()
    q_table = env.get_qtable("q_table_1639638457.pickle")
    episode_rewards = []
    obs = env.reset()
    done = False
    episode_reward=0
    episode=1
    t=200
    while not done:
        print(f'episode #{episode}, episode_reward:{episode_reward}')
        action = np.argmax(q_table[obs])
        new_obs, reward, done = env.step(action)
        obs=new_obs
        episode_reward += reward
        episode+=1
        env.render(t)
    print(f'All_episode #{episode}, ALL_episode_reward:{episode_reward},我真棒yyds')
    episode_rewards.append(episode_reward)

if __name__ == '__main__':
   # train()
   test()
