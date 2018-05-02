#!/usr/bin/env python

"""
Code to load an expert policy and generate roll-out data for behavioral cloning.
Example usage:
    python run_expert_bc.py experts/Humanoid-v1.pkl Humanoid-v1 --render \
            --num_rollouts 20

Author of this script and included expert policies: Jonathan Ho (hoj@openai.com)
"""

import pickle
import tensorflow as tf
import numpy as np
import tf_util
import gym
import load_policy
import math
import matplotlib.pyplot as plt
import argparse


class GymData:
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def set_batch_size(self, batch_size):
        self.current_batch = 0
        self.batch_size = batch_size
        self.num_batches = math.ceil(len(self.data) / self.batch_size)

    def next_batch(self):
        if self.current_batch == self.num_batches:
            self.current_batch = 0
            # shuffle
            self.shuffle()
            return None
        else:
            begin = self.current_batch * self.batch_size
            finish = (self.current_batch + 1) * self.batch_size

            if finish > len(self.data):
                finish = len(self.data)

            data_to_return = self.data[begin: finish, :]
            labels_to_return = self.labels[begin: finish, :]
            self.current_batch += 1
            return data_to_return, labels_to_return

    def shuffle(self):
        idx = np.arange(0, len(self.data))
        np.random.shuffle(idx)
        self.data = self.data[idx, :]
        self.labels = self.labels[idx, :]


def mlp(num_inputs,num_outputs):
    # Parameters
    learning_rate = 0.01
    training_epochs = 1000

    # Network Parameters
    n_hidden_1 = 64  # 1st layer number of neurons
    n_hidden_2 = 64  # 2nd layer number of neurons
    n_input = num_inputs
    n_output = num_outputs

    X = tf.placeholder("float", [None, n_input])
    Y = tf.placeholder("float", [None, n_output])

    # weights = {
    #     'h1': tf.get_variable("h1", shape=[n_input, n_hidden_1],
    #        initializer=tf.contrib.layers.xavier_initializer()),
    #     'h2': tf.get_variable("h2", shape=[n_hidden_1, n_hidden_2],
    #        initializer=tf.contrib.layers.xavier_initializer()),
    #     'out': tf.get_variable("out", shape=[n_hidden_2, n_output],
    #        initializer=tf.contrib.layers.xavier_initializer())
    # }
    # biases = {
    #     'b1': tf.get_variable("b1", shape=[n_hidden_1],
    #        initializer=tf.contrib.layers.xavier_initializer()),
    #     'b2': tf.get_variable("b2", shape=[n_hidden_2],
    #        initializer=tf.contrib.layers.xavier_initializer()),
    #     'out': tf.get_variable("bout", shape=[n_output],
    #        initializer=tf.contrib.layers.xavier_initializer())
    # }

    # Store layers weight & bias
    weights = {
        'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
        'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
        'out': tf.Variable(tf.random_normal([n_hidden_2, n_output]))
    }
    biases = {
        'b1': tf.Variable(tf.random_normal([n_hidden_1])),
        'b2': tf.Variable(tf.random_normal([n_hidden_2])),
        'out': tf.Variable(tf.random_normal([n_output]))
    }

    # Hidden fully connected layer with 256 neurons
    layer_1 = tf.nn.tanh(tf.add(tf.matmul(X, weights['h1']), biases['b1']))
    # Hidden fully connected layer with 256 neurons
    layer_2 = tf.nn.tanh(tf.add(tf.matmul(layer_1, weights['h2']), biases['b2']))
    # Output fully connected layer with a neuron for each class
    pred = tf.add(tf.matmul(layer_2, weights['out']), biases['out'], name="predictions")

    # Define loss and optimizer
    # learning objective : minimize mean squared loss between predicted action and actual action (mean l2 loss)
    # loss_op = tf.reduce_mean(tf.pow((pred - Y), 2), name="loss")
    loss_op = tf.reduce_mean(tf.losses.mean_squared_error(pred, Y),name="loss")
    # our optimizer is declared in this step - adam
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    # In this op we are actually training. Sending the loss to optimizer and asking it to minimize that loss
    train_op = optimizer.minimize(loss_op)
    # Initializing the variables
    init = tf.global_variables_initializer()

    return {
        "init": init,
        "training_epochs": training_epochs,
        "loss_op": loss_op,
        "train_op": train_op,
        "X": X,
        "Y": Y,
        "predictions": pred
    }


def inference(policy_fn, network, env, sess, num_rollouts, max_steps, is_render=False):
    returns = []
    observations = []
    actions = []
    expert_actions = []
    for i in range(num_rollouts):
        if i%50 == 0:
            print('iter', i)
        obs = env.reset()
        obs = obs.reshape(1, -1)
        done = False
        totalr = 0.
        steps = 0
        while not done:
            action = predict_nn_output(network, obs, sess)

            observations.append(obs)
            actions.append(action)

            obs, r, done, _ = env.step(action)
            if done:
                print("Done",steps)
            obs = obs.reshape(1, -1)
            totalr += r
            steps += 1
            if is_render:
                env.render()
            # if steps % 100 == 0:
            #     print("%i/%i" % (steps, max_steps))
            if steps >= max_steps:
                break
        returns.append(totalr)

    print("Getting the expert actions")
    with tf.Session():
        for obs in observations:
            expert_action = policy_fn(obs)
            expert_actions.append(expert_action)
    print("Done with the expert actions")
    print('nn_returns', returns)
    print('nn_mean return', np.mean(returns))
    print('std of return', np.std(returns))
    # sess.close()
    return {'observations': np.array(observations),
                       'actions': np.array(expert_actions)}


def predict_nn_output(network, obs, sess):
    action = sess.run(network["predictions"], feed_dict={network["X"]: obs})
    return action

def generate_rollout(args,policy_fn):
    with tf.Session():
        tf_util.initialize()

        env = gym.make(args.envname)
        max_steps = args.max_timesteps or env.spec.timestep_limit

        returns = []
        observations = []
        actions = []
        for i in range(args.num_rollouts):
            # print('iter', i)
            obs = env.reset()
            done = False
            totalr = 0.
            steps = 0
            while not done:
                action = policy_fn(obs[None, :])
                observations.append(obs)
                actions.append(action)
                obs, r, done, _ = env.step(action)
                totalr += r
                steps += 1
                if args.render:
                    env.render()
                # if steps % 100 == 0: print("%i/%i" % (steps, max_steps))
                if steps >= max_steps:
                    break
            returns.append(totalr)

        print('returns', returns)
        print('mean return', np.mean(returns))
        print('std of return', np.std(returns))

        return {'observations': np.array(observations),
                       'actions': np.array(actions)}

def plot(loss_over_epochs):
    fig = plt.figure(figsize=(10,8))
    plt.plot(loss_over_epochs)
    plt.ylim(0,100000)
    plt.xlabel("Number of epochs")
    plt.ylabel("Training loss")
    plt.savefig("warmupPlot_mb.png")


def train(expert_data,is_plot=False):
    num_inputs = expert_data['observations'].shape[1]
    num_outputs = expert_data['actions'].shape[1]
    sess = tf.Session()

    train_X = expert_data['observations']
    train_Y = expert_data['actions'].reshape(-1, num_outputs)  # (50k,1,3) --> (50k,3)
    g_data = GymData(train_X,train_Y)
    g_data.set_batch_size(1000)
    g_data.shuffle()

    nn = mlp(num_inputs,num_outputs)
    # 1st step: Global Variable Initialization
    sess.run(nn["init"])

    loss_over_epochs = []
    for epoch in range(nn["training_epochs"]):
        cost = 0
        batch = g_data.next_batch()
        while batch is not None:
            data_x = batch[0]
            data_y = batch[1]
            sess.run(nn["train_op"], feed_dict={nn["X"]: data_x, nn["Y"]: data_y})
            cost += sess.run(nn["loss_op"], feed_dict={nn["X"]: data_x, nn["Y"]: data_y})
            batch = g_data.next_batch()
        loss_over_epochs.append(cost)
        if epoch%200 == 0:
            print(cost,epoch)
    if is_plot:
        plot(loss_over_epochs)
    return nn,sess

def aggregator(data1, data2):
    print("Aggregating")
    n_inputs = data1['observations'].shape[1]
    observations = np.vstack([data1['observations'],data2['observations'].reshape(-1,n_inputs)])
    actions = np.vstack([data1['actions'],data2['actions']])

    return {'observations': observations,
            'actions': actions}



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('expert_policy_file', type=str)  # --> pkl file
    parser.add_argument('envname', type=str)  # Hopper-v1
    parser.add_argument('--render', action='store_true')
    parser.add_argument("--max_timesteps", type=int)
    parser.add_argument('--num_rollouts', type=int, default=20,
                        help='Number of expert roll outs')
    args = parser.parse_args()
    env = gym.make(args.envname)
    max_steps = args.max_timesteps or env.spec.timestep_limit
    print('loading and building expert policy')
    policy_fn = load_policy.load_policy(args.expert_policy_file)
    print('loaded and built')


    expert_data = generate_rollout(args,policy_fn)

    print(expert_data['observations'].shape) #--(20000, 11)
    print(expert_data['actions'].shape) # --(20000, 1, 3)

    dagger_steps=10
    for i in range(dagger_steps):
        nn,sess= train(expert_data)
        # new_expert_data: contains new observations and experts' actions
        new_expert_data = inference(policy_fn, nn, env, sess, args.num_rollouts, max_steps)
        expert_data = aggregator(expert_data,new_expert_data)
        print("Dagger step",i)



if __name__ == '__main__':
    main()
