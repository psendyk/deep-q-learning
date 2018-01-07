"""Reference: https://www.nature.com/articles/nature14236"""

import numpy as np
import matplotlib.pyplot as plt
import gym
from keras.models import Sequential
from keras.layers import Flatten, Conv2D, Dense, Activation
import random

#Set up the environment
env = gym.make('CartPole-v0')
state_space = 4
action_space = 2

#Create the network
def make_model(m=None):
    model = Sequential()
    model.add(Dense(10, input_shape = (state_space,)))
    model.add(Activation('relu'))
    model.add(Dense(5))
    model.add(Activation('relu'))
    model.add(Dense(action_space))
    model.compile(loss='mse', optimizer='adam')
    if m:
        model.set_weights(m.get_weights())
    return model

model = make_model()
tmodel = make_model(model)
update_target = 5

#Training parameters
num_episodes = 1000
epsilon = 0.25
anneal = 0.0025
exp_buffer = []
batch_size = 100
gamma = 0.99

#Training
for i in range(num_episodes):
    obs = env.reset().reshape((1,state_space))
    done = False
    total_reward = 0

    while not done:
        if random.random() < epsilon:
            action = random.randint(0, action_space-1)
        else:
            q_values = model.predict(obs)
            action = np.argmax(q_values)
        obs1, reward, done, _ = env.step(action)
        obs1 = obs1.reshape((1,state_space)) 
        total_reward += reward
        exp_buffer.append((obs,action,reward,obs1,done))
        obs = obs1

        if i % update_target == 0:
            tmodel.set_weights(model.get_weights())

    if len(exp_buffer) > batch_size:
        minibatch = random.sample(exp_buffer, batch_size)
        inputs = []
        q_values = []
        for m in minibatch:
            obs, action, reward, obs1, done = m
            inputs.append(obs)
            q_vals = tmodel.predict(obs)
            if done:
                q_vals[0][action] = reward
            else:
                q_vals[0][action] = reward + gamma * np.max(tmodel.predict(obs))
            q_values.append(q_vals)
        inputs = np.array(inputs).reshape((batch_size, state_space))
        q_values = np.array(q_values).reshape((batch_size, action_space))
        model.fit(inputs, q_values, verbose=False)

    epsilon -= anneal
    print(total_reward)
