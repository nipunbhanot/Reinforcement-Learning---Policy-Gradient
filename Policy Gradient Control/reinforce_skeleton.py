from grid_world import *
from numpy.random import choice
import numpy as np
import random
from matplotlib import pyplot as plt

class DiscreteSoftmaxPolicy(object):
    def __init__(self, num_states, num_actions, temperature):
        self.num_states = num_states
        self.num_actions = num_actions
        self.temperature = temperature
        self.init_temperature = temperature
        self.terminal_temperature = 0.8

        # here are the weights for the policy
        self.weights = np.random.randn(num_states, num_actions) * np.sqrt(2/num_actions)
        

    # TODO: fill this function in
    # it should take in an environment state
    # return the action that follows the policy's distribution
    def act(self, state):
        probs = self._action_probs(state)
        return choice(range(self.num_actions), p=probs)

    def _sigmoid(self):
        return 1 / (1 + np.exp(-self.weights))

    def _action_probs(self, state):
        prob_actions = np.zeros((self.num_actions, ))
        total = 0
        for a in range(self.num_actions):
            prob_actions[a] = np.exp( self.weights[state][a] / self.temperature )
            total += prob_actions[a]
        
        return prob_actions / total

    def _custom_softmax(self, weights, state):
        prob_actions = np.zeros((self.num_actions, ))
        total = 0
        for a in range(self.num_actions):
            prob_actions[a] = np.exp( weights[state][a] / self.temperature )
            total += prob_actions[a]
        
        return prob_actions / total

    # TODO: fill this function in
    # computes the gradient of the discounted return 
    # at a specific state and action
    # return the gradient, a (self.num_states, self.num_actions) numpy array
    def compute_gradient(self, state, action, discounted_return):
        grad = np.zeros((self.num_states, self.num_actions))
        probs = self._action_probs(state)

        for a in range(self.num_actions):
            if a == action:
                grad[state][a] = ( self.weights[state][action] / self.temperature ) * ( 1 - probs[a] ) * discounted_return

            else:
                grad[state][a] = ( self.weights[state][action] / self.temperature ) * (- probs[a] ) * discounted_return

        return grad


    # TODO: fill this function in
    # takes a step of gradient ascent given a gradient (from compute_gradient())
    # and a step size. adjust self.weights
    def gradient_step(self, grad, step_size):
        self.weights = self.weights + grad * step_size


# TODO: fill this function in
# takes in a list of rewards from an episode
# and returns a list of discounted rewards
# Ex. get_discounted_returns([1, 1, 1], 0.5)
# should return [1.75, 1.5, 1]
def get_discounted_returns(rewards, gamma):
    cumulative_returns = np.zeros((len(rewards), ))
    future_returns = 0
    for i in range(len(rewards) - 1, -1, -1):
        cumulative_returns[i] = rewards[i] + gamma * future_returns
        future_returns = cumulative_returns[i]
    return cumulative_returns


# TODO: fill this function in 
# this will take in an environment, GridWorld
# a policy (DiscreteSoftmaxPolicy)
# a discount rate, gamma
# and the number of episodes you want to run the algorithm for
def reinforce(env, policy, gamma, num_episodes, learning_rate):
    total_rewards = []
    state = env.reset() # initial state
    for ep in range(num_episodes):
        # Decrease temperature overtime
        policy.temperature -= ((policy.init_temperature - policy.terminal_temperature) / num_episodes)
        rewards = []
        states = []
        actions = []

        done = False

        # Generate a full episode folowing the policy
        while not done:
            states.append(state)

            action = policy.act(state)
            actions.append(action)

            state, reward, done = env.step(action)
            rewards.append(reward)
        state = env.reset()

        G = get_discounted_returns(rewards, gamma)

        # loop through each episode
        for i in range(len(states)):          
            grad = policy.compute_gradient(states[i], actions[i], G[i])
            policy.gradient_step(grad, learning_rate)
        total_rewards.append(np.mean(rewards))
    total_rewards = np.convolve(total_rewards, np.ones((300,)) / 300, mode='valid')
    return list(range(len(total_rewards))), total_rewards

def print_policy(policy: DiscreteSoftmaxPolicy, env: GridWorld, i):
        result = np.chararray((env.n_rows, env.n_cols), unicode=True)
        act_arrow_map = {
            0: u"\u2191",
            1: u"\u2192",
            2: u"\u2193",
            3: u"\u2190"
        }
        for r in range(env.n_rows):
            for c in range(env.n_cols):
                result[r][c] = act_arrow_map[policy.act(env.map[r][c])]

        print('Iteration: [{}]\nPolicy:\n{}\n\n'.format(i, result))


if __name__ == "__main__":
    gamma = 0.9
    num_episodes = 20000
    learning_rate = 1e-4

    fig = plt.figure()
    for i in range(1, 21):
        env = GridWorld(MAP2)
        policy = DiscreteSoftmaxPolicy(env.get_num_states(), env.get_num_actions(), temperature=1.4)
        iteration, total_mean_return = reinforce(env, policy, gamma, num_episodes, learning_rate)
        
        print_policy(policy, env, i)

        ax = fig.add_subplot(4,5,i)
        ax.plot(iteration, total_mean_return)
        ax.set_title(i)
        print('{} | Reached goal: {} times\n\n'.format(i, env.goals_reached))

    plt.tight_layout()
    plt.show()


    # gives a sample of what the final policy looks like
    print("Rolling out final policy")
    state = env.reset()
    env.print()
    done = False
    while not done:
        input("press enter to continue:")
        action = policy.act(state)
        state, reward, done = env.step(action)
        env.print()
