# -*- coding: utf-8 -*-
"""
@author: Olivier Sigaud

A merge between two sources:

* Adaptation of the MountainCar Environment from the "FAReinforcement" library
of Jose Antonio Martin H. (version 1.0), adapted by  'Tom Schaul, tom@idsia.ch'
and then modified by Arnaud de Broissia

* the OpenAI/gym MountainCar environment
itself from
http://incompleteideas.net/sutton/MountainCar/MountainCar1.cp
permalink: https://perma.cc/6Z2N-PFWC
"""

import math

import numpy as np
from itertools import product
POSITIONS = np.linspace(-1.2, 0.6, 200)
VELOCITIES = np.linspace(-0.07, 0.07, 200)
ACTIONS = np.linspace(-1, 1, 10)


class Continuous_MountainCarEnv:

    def __init__(self, goal_velocity = 0):
        self.min_action = -1.0
        self.max_action = 1.0
        self.min_position = -1.2
        self.max_position = 0.6
        self.max_speed = 0.07
        self.goal_position = 0.45 # was 0.5 in gym, 0.45 in Arnaud de Broissia's version
        self.goal_velocity = goal_velocity
        self.power = 0.0015
        self.step_count = 0
        self.max_step_count = 999

        self.low_state = np.array([self.min_position, -self.max_speed])
        self.high_state = np.array([self.max_position, self.max_speed])

        self.seed()
        self.reset()

    def seed(self, seed=None):
        np.random.seed(seed)
    
    def step(self, action):
        position = POSITIONS[self.state[0]]
        velocity = VELOCITIES[self.state[1]]
        action_val = ACTIONS[action[0]]
        force = min(max(action_val, -1.0), 1.0)

        velocity += force*self.power -0.0025 * math.cos(3*position)
        if (velocity > self.max_speed): velocity = self.max_speed
        if (velocity < -self.max_speed): velocity = -self.max_speed
        position += velocity
        if (position > self.max_position): position = self.max_position
        if (position < self.min_position): position = self.min_position
        if (position==self.min_position and velocity<0): velocity = 0

        done = bool((position >= self.goal_position and velocity >= self.goal_velocity) or self.step_count > self.max_step_count)

        reward = 0
        if done:
            reward = 100.0
        reward-= math.pow(action_val,2)*0.1

        self.state = self.find_bucket([position, velocity])
        # self.state = np.array([position, velocity])
        self.step_count += 1
        return self.state, reward, done, {}

    def reset(self):
        self.state = np.array([0,0])
        self.step_count = 0
        return np.array(self.state)

    def print(self):
        print(self.state)

    def find_bucket(self, sample):
        # Finds the corresponding discrete bucket for the given state
        position, velocity = sample[0], sample[1]
        state = np.array([0] * 2)
        state[0] = int((position - self.low_state[0]) / (self.high_state[0] - self.low_state[0]) * (len(POSITIONS) - 1))
        state[1] = int((velocity - self.low_state[1]) / (self.high_state[1] - self.low_state[1]) * (len(VELOCITIES) - 1))

        return state