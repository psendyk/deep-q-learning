import numpy as np
import matplotlib.pyplot as plt
import cv2
import gym
from keras.models import Sequential
from keras.layers import Flatten, Conv2D, Dense, Activation
import random

def preprocess(obs):
    obs = np.array(obs, dtype=np.uint8)
    obs = cv2.resize(obs, (84, 110))
    obs = cv2.cvtColor(obs, cv2.COLOR_BGR2GRAY)
    obs = obs[26:110,:]
    _, obs = cv2.threshold(obs, 1, 255, cv2.THRESH_BINARY)
    return np.reshape(obs, (1, 84, 84, 1)) / 255

#Set up the environment
env = gym.make('SpaceInvaders-v0')
state_space = (84, 84, 1)
action_space = 6

#Create the network
model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(8, 8), strides=4, input_shape=state_space))
model.add(Activation('relu'))
model.add(Conv2D(filters=64, kernel_size=(4,4), strides=2))
model.add(Activation('relu'))
model.add(Conv2D(filters=64, kernel_size=(3,3), strides=1))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(action_space))
model.compile(loss='mse', optimizer='adam')

#Training parameters
num_episodes = 1000
epsilon = 0.25
anneal = 0.0025
exp_buffer = []
batch_size = 100
gamma = 0.99

#Training
for i in range(num_episodes):
    obs = env.reset()
    obs = preprocess(obs)
    done = False
    total_reward = 0

    while not done:
        if random.random() < epsilon:
            action = random.randint(0, action_space-1)
        else:
            q_values = model.predict(obs)
            action = np.argmax(q_values)
        obs1, reward, done, _ = env.step(action)
        obs1 = preprocess(obs1)
        total_reward += reward
        exp_buffer.append((obs,action,reward,obs1,done))
        obs = obs1

    if len(exp_buffer) > batch_size:
        minibatch = random.sample(exp_buffer, batch_size)
        inputs = []
        q_values = []
        for m in minibatch:
            obs, action, reward, obs1, done = m
            inputs.append(obs)
            q_vals = model.predict(obs)
            if done:
                q_vals[0][action] = reward
            else:
                q_vals[0][action] = reward + gamma * np.max(model.predict(obs))
            q_values.append(q_vals)
        inputs = np.array(inputs).reshape((batch_size, 84, 84, 1))
        q_values = np.array(q_values).reshape((batch_size, action_space))
        model.fit(inputs, q_values, verbose=False)

    epsilon -= anneal
    print(total_reward)
