'''
parse
get session
mlp
run cloning
get reward

'''



import tensorflow as tf
import numpy as np
import tf_util
import gym
import load_policy
import math
import matplotlib.pyplot as plt
import argparse
from run_expert import get_expert_data


def get_tf_session():
    """ Returning a session. """
    tf.reset_default_graph()
    session = tf.Session()
    return session


def get_minibatch(expert_data, mb_size=128):
    observations = expert_data["observations"]
    actions = expert_data["actions"]
    indices = np.arange(observations.shape[0])
    np.random.shuffle(indices)
    mb_observations = observations[indices[:mb_size], :]
    mb_actions = actions[indices[:mb_size], :].squeeze()
    return mb_observations, mb_actions


def mlp(input_data, num_out):
    num_hidden_1 = 128
    num_hidden_2 = 128
    with tf.variable_scope("clone_net", reuse=False):
        layer_1 = tf.contrib.layers.fully_connected(input_data, num_outputs=num_hidden_1,
                                                weights_initializer=tf.contrib.layers.xavier_initializer(uniform=True),
                                                activation_fn=tf.nn.tanh)
        layer_2 = tf.contrib.layers.fully_connected(layer_1, num_outputs=num_hidden_2,
                                                weights_initializer=tf.contrib.layers.xavier_initializer(uniform=True),
                                                activation_fn=tf.nn.tanh)
        out = tf.contrib.layers.fully_connected(layer_2, num_outputs=num_out,
                                                weights_initializer=tf.contrib.layers.xavier_initializer(uniform=True),
                                                activation_fn=None)
        return out


def run_cloning(args,session):
    env = gym.make(args.envname)
    expert_data,_ = get_expert_data(args.expert_policy_file, args.envname, args.max_timesteps, args.num_rollouts)
    print("Got rollouts from expert.")
    expert_obs = expert_data["observations"]
    expert_acts = expert_data["actions"]
    x = tf.placeholder(tf.float32, shape=[None, expert_obs.shape[-1]])
    y = tf.placeholder(tf.float32, shape=[None, expert_acts.shape[-1]])
    nn_policy = mlp(x, expert_acts.shape[-1])

    # # Save weights as a single vector to make saving/loading easy.
    # weights_bc = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='clone_net')
    # weight_vector = tf.concat([tf.reshape(w, [-1]) for w in weights_bc], axis=0)

    # Construct the loss function and training information.
    l2_loss = tf.reduce_mean(
        tf.reduce_sum((nn_policy - y) * (nn_policy - y), axis=[1])
    )
    train_step = tf.train.AdamOptimizer(args.learning_rate).minimize(l2_loss)

    loss_over_epochs = []
    epochs = []
    returns_over_epochs = []
    session.run(tf.global_variables_initializer())

    for i in range(args.training_epochs):
        mb_obs,mb_acts = get_minibatch(expert_data,args.minibatch_size)
        _, training_loss = session.run([train_step, l2_loss], feed_dict={x: mb_obs, y: mb_acts})

        if (i % 200 == 0):
            returns = get_rewards(args, session, nn_policy, x, env)
            print("Iteration :",i, "Training Loss", training_loss)
            print("mean(returns): {}\nstd(returns): {}\n".format(np.mean(returns), np.std(returns)))
            epochs.append(i)
            loss_over_epochs.append(training_loss)
            returns_over_epochs.append(returns)
    mean_return_over_epochs = list(map(lambda x:np.mean(x),returns_over_epochs))
    plot(epochs, loss_over_epochs, mean_return_over_epochs)

def dagger(args,session):
    env = gym.make(args.envname)
    expert_data, expert_returns = get_expert_data(args.expert_policy_file, args.envname, args.max_timesteps, args.num_rollouts)
    print("Got rollouts from expert.")
    expert_obs = expert_data["observations"]
    expert_acts = expert_data["actions"]
    policy_fn = load_policy.load_policy(args.expert_policy_file)


    x = tf.placeholder(tf.float32, shape=[None, expert_obs.shape[-1]])
    y = tf.placeholder(tf.float32, shape=[None, expert_acts.shape[-1]])
    nn_policy = mlp(x, expert_acts.shape[-1])
    l2_loss = tf.reduce_mean(
        tf.reduce_sum((nn_policy - y) * (nn_policy - y), axis=[1])
    )
    train_step = tf.train.AdamOptimizer(args.learning_rate).minimize(l2_loss)

    loss_over_epochs = []
    epochs = []
    returns_over_epochs = []
    session.run(tf.global_variables_initializer())

    dagger_steps = []
    dagger_returns =[]
    for step in range(args.dagger_steps):
        print("running dagger step", step)

        for i in range(args.training_epochs):
            mb_obs, mb_acts = get_minibatch(expert_data, args.minibatch_size)
            _, training_loss = session.run([train_step, l2_loss], feed_dict={x: mb_obs, y: mb_acts})

            if (i % args.check_every == 0):
                returns = get_rewards(args, session, nn_policy, x, env)
                print("Iteration :", i, "Training Loss", training_loss)

                print("mean(returns): {}\nstd(returns): {}\n".format(
                    np.mean(returns), np.std(returns)))
                epochs.append(i)
                loss_over_epochs.append(training_loss)
                returns_over_epochs.append(returns)

        # Adding
        new_obs, new_actions = get_expert_labels(args, session, nn_policy, policy_fn, x, env)
        expert_data['observations']=np.vstack([expert_data['observations'],new_obs])
        expert_data['actions']=np.vstack([expert_data['actions'],new_actions])
        dagger_steps.append(step)
        dagger_returns.append([np.mean(returns), np.std(returns)])

    plot_dagger(dagger_steps,dagger_returns,expert_returns)

def get_expert_labels(args, session, nn_policy, policy_fn, x, env):
    expert_actions = []
    observations = []
    returns = []
    max_steps = env.spec.timestep_limit

    for _ in range(args.num_rollouts):
        obs = env.reset()
        done = False
        totalr = 0
        steps = 0
        while not done:
            # Take steps by expanding observation (to get shapes to match).
            exp_obs = np.expand_dims(obs, axis=0)
            action = np.squeeze(session.run(nn_policy, feed_dict={x: exp_obs}))
            obs, r, done, _ = env.step(action)
            totalr += r
            observations.append(obs)
            steps += 1
            if args.render: env.render()
            if steps >= max_steps: break
        returns.append(totalr)
    with tf.Session():
        for obs in observations:
            expert_actions.append(policy_fn(obs[None, :]))

    return np.array(observations), np.array(expert_actions)



def get_rewards(args, session, nn_policy, x, env):
    actions = []
    observations = []
    returns = []
    max_steps = env.spec.timestep_limit

    for _ in range(args.num_rollouts):
        obs = env.reset()
        done = False
        totalr = 0
        steps = 0
        while not done:
            # Take steps by expanding observation (to get shapes to match).
            exp_obs = np.expand_dims(obs, axis=0)
            action = np.squeeze(session.run(nn_policy, feed_dict={x: exp_obs}))
            obs, r, done, _ = env.step(action)
            totalr += r
            steps += 1
            if args.render: env.render()
            if steps >= max_steps: break
        returns.append(totalr)
        # return the total reward

    return returns

def plot(epochs,loss_over_epochs,returns_over_epochs):
    fig = plt.figure(figsize=(10,8))
    plt.plot(epochs,loss_over_epochs,'r')
    plt.xlabel("Number of epochs")
    plt.ylabel("Training loss")
    plt.savefig("LossHalfCheetah.png")

    fig = plt.figure(figsize=(10, 8))
    plt.plot(epochs, returns_over_epochs, 'b')
    plt.xlabel("Number of epochs")
    plt.ylabel("Returns")
    plt.savefig("ReturnsHalfCheetah.png")

def plot_dagger(dagger_steps, dagger_returns, expert_returns):
    dagger_returns = np.array(dagger_returns)
    expert_returns = np.array(len(dagger_steps) * [expert_returns])
    bc_returns = np.array(len(dagger_steps) * [dagger_returns[0]])
    fig = plt.figure(figsize=(10,8))
    plt.errorbar(dagger_steps, dagger_returns[:,0], dagger_returns[:,1],color='r',fmt='-.',label="Dagger")
    plt.errorbar(dagger_steps, expert_returns[:,0], expert_returns[:,1], color='g',fmt='-.', label="Expert")
    plt.errorbar(dagger_steps, bc_returns[:,0], bc_returns[:,1], color='b',fmt='-.', label="BC")
    plt.legend()
    print(expert_returns[:,1])
    plt.xlabel("Number of Dagger Iterations")
    plt.ylabel("Performance")
    plt.savefig("Dagger_plot_ant.png")




def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('expert_policy_file', type=str)
    parser.add_argument('envname', type=str)
    parser.add_argument('--render', action='store_true')
    parser.add_argument("--max_timesteps", type=int)
    parser.add_argument('--num_rollouts', type=int, default=25,
                        help='Number of expert roll outs')
    parser.add_argument('--training_epochs', type=int, default=5001,
                        help='Number of training epochs')
    parser.add_argument('--minibatch_size', type=int, default=10000,
                        help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning Rate')
    parser.add_argument('--check_every', type=int, default=5000,
                        help='Check Performance in this many epochs')

    parser.add_argument('--dagger_steps', type=int, default=10,
                        help='Number of dagger steps')
    parser.add_argument('--run_type', type=str, default="bc",
                        help="bc: Behavior cloning, dag: Dagger")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    session = get_tf_session()
    # Seed the generator
    np.random.seed(7)
    tf.set_random_seed(11)

    if args.run_type=="bc":
        run_cloning(args,session)
    elif args.run_type=="dag":
        dagger(args,session)
    else:
        print("Select a valid run type : bc: Behavior cloning, dag: Dagger")

if __name__ == "__main__":
    main()
