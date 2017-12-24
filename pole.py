import numpy as np
import random
import tensorflow as tf
import gym
import tempfile

#Environment
env = gym.make('CartPole-v0')
actions = 2
state_space_dims = 4

#Q-learning parameters
num_episodes = 1000
epsilon = 0.5
anneal = 0.005
exp_buffer = []
batch_size = 100
gamma = 0.99

#Creating the network
x = tf.placeholder(tf.float32, shape=[None, state_space_dims])
y_ = tf.placeholder(tf.float32, shape=[None, actions])

W1 = tf.Variable(tf.random_normal([state_space_dims, 10], stddev=0.1), name="W1")
b1 = tf.Variable(tf.random_normal([10]), name='b1')
h_fc1 = tf.nn.relu(tf.add(tf.matmul(x, W1), b1))

W2 = tf.Variable(tf.random_normal([10, 5], stddev=0.1), name="W2")
b2 = tf.Variable(tf.random_normal([5]), name='b2')
h_fc2 = tf.nn.relu(tf.add(tf.matmul(h_fc1, W2), b2))

W3 = tf.Variable(tf.random_normal([5, actions], stddev=0.1), name="W3")
b3 = tf.Variable(tf.random_normal([actions]), name='b3')
y_pred = tf.add(tf.matmul(h_fc2, W3), b3)

with tf.name_scope('loss'):
    mse = tf.losses.mean_squared_error(labels=y_, predictions=y_pred)
with tf.name_scope('adam-optimizer'):
    adam = tf.train.AdamOptimizer().minimize(mse)
with tf.name_scope('accuracy'):
    correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_, 1))
    correct_prediction = tf.cast(correct_prediction, tf.float32)
accuracy = tf.reduce_mean(correct_prediction)

graph_location = tempfile.mkdtemp()
print('Saving graph to: %s' % graph_location)
train_writer = tf.summary.FileWriter(graph_location)
train_writer.add_graph(tf.get_default_graph())

#Training
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(num_episodes):
        obs = env.reset().reshape((1, state_space_dims))
        done = False
        total_reward = 0
        while not done:
            if random.random() < epsilon:
                action = random.randint(0, actions - 1)
            else:
                q_values = sess.run(y_pred, feed_dict={x: obs})
                action = np.argmax(q_values)

            obs1, reward, done, _ = env.step(action)
            obs1 = obs1.reshape((1, state_space_dims))
            total_reward += reward
            exp_buffer.append((obs, action, reward, obs1, done))
            obs = obs1

        if len(exp_buffer) > batch_size:
            minibatch = random.sample(exp_buffer, batch_size)
            inputs = []
            q_values = []
            for m in minibatch:
                obs, action, reward, obs1, done = m
                inputs.append(obs)
                q_vals = sess.run(y_pred, feed_dict={x: obs})
                if done:
                    q_vals[0][action] = reward
                else:
                    q_vals[0][action] = reward + gamma * np.max(sess.run(y_pred, feed_dict={x: obs1}))
                q_values.append(q_vals)
            inputs = np.array(inputs).reshape(batch_size, state_space_dims)
            q_values = np.array(q_values).reshape(batch_size, actions)
            sess.run(adam, feed_dict={x: inputs, y_: q_values})

        epsilon -= anneal
        print(total_reward)
	    
