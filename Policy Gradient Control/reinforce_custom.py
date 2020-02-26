from mountain_car_discrete import *
import numpy as np
import random
from matplotlib import pyplot as plt

class CustomPolicy(object):
    def __init__(self, num_states, num_actions, epsilon = 0.9):
        self.num_states = num_states
        self.num_actions = num_actions

        # here are the weights for the policy - you may change this initialization
        self.weights = np.random.randn(self.num_states, self.num_actions) * np.sqrt(2/self.num_actions)
        self.epsilon = epsilon
        self.init_epsilon = epsilon
        self.min_epsilon = 0.1
        

    # TODO: fill this function in
    # it should take in an environment state
    # return the action that follows the policy's distribution
    def act(self, state):
        probs = self._action_prob(state)
        if random.random() < self.epsilon:
            action = np.random.choice(self.num_actions)
        else:
            action = np.argmax(probs)

        return [action]

    def _action_prob(self, state):
        # loop through all actions
        prob_actions = np.zeros((self.num_actions, ))
        total = 0
        for a in range(self.num_actions):
            prob_actions[a] = np.exp( state.dot(self.weights)[a] )

            total += prob_actions[a]
        
        return prob_actions / total

    # TODO: fill this function in
    # computes the gradient of the discounted return 
    # at a specific state and action
    # return the gradient, a (self.num_states, self.num_actions) numpy array
    def compute_gradient(self, state, action, discounted_return):        
        grad = np.zeros((self.num_states, self.num_actions))
        probs = self._action_prob(state)

        # grad - num_states x num_actions
        for a in range(self.num_actions):
            if a == action:
                grad[0][a] = self.weights[0][action[0]] * ( 1 - probs[a] ) * discounted_return
                grad[1][a] = self.weights[1][action[0]] * ( 1 - probs[a] ) * discounted_return
                
            else:
                grad[0][a] = self.weights[0][action[0]] * ( - probs[a] ) * discounted_return
                grad[1][a] = self.weights[1][action[0]] * ( - probs[a] ) * discounted_return

        return grad


    # TODO: fill this function in
    # takes a step of gradient ascent given a gradient (from compute_gradient())
    # and a step size. adjust self.weights
    def gradient_step(self, grad, step_size):
        self.weights = np.add(self.weights, step_size * grad)


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
        rewards = []
        states = []
        actions = []

        done = False
        policy.epsilon -= (policy.init_epsilon - policy.min_epsilon) / num_episodes

        # Generate a full episode folowing the policy
        while not done:
            states.append(state)

            action = policy.act(state)
            actions.append(action)

            state, reward, done, _ = env.step(action)
            rewards.append(reward)
        state = env.reset()

        G = get_discounted_returns(rewards, gamma)

        # loop through each episode
        for i in range(len(states)):          
            grad = policy.compute_gradient(states[i], actions[i], G[i] * gamma)
            policy.gradient_step(grad, learning_rate)
        total_rewards.append(np.sum(rewards))
        if ep % 100 == 0:
            print('EP{}: {}'.format(ep, total_rewards[-1]))
    print('Min: {} | Max: {}'.format(np.min(total_rewards), np.max(total_rewards))) # min of 20k episodes, 1/5 training steps
    total_rewards = np.convolve(total_rewards, np.ones((30,)) / 30, mode='valid')
    plt.plot(list(range(len(total_rewards))), total_rewards)
    plt.show()


if __name__ == "__main__":
    gamma = 0.95
    num_episodes = 10000
    learning_rate = 1e-4
    epsilon = 0.5
    env = Continuous_MountainCarEnv()

    possible_actions = np.linspace(-1, 1, 10)
    num_actions = len(possible_actions)

    policy = CustomPolicy(2, num_actions, epsilon)
    reinforce(env, policy, gamma, num_episodes, learning_rate)

    # gives a sample of what the final policy looks like
    print("Rolling out final policy")
    policy.epsilon = 0 # take argmax
    state = env.reset()
    env.print()
    done = False
    while not done:
        input("press enter to continue:")
        action = policy.act(state)
        state, reward, done, _ = env.step([action])
        env.print()
