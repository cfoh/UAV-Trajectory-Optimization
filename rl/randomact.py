'''
This module contains class implementing random action.
A valid random action will be drawn when requested.
'''

from rl.rlbase import RL
import random

class RandomAction(RL):

    def __init__(self):
        super().__init__(name="Random Action")

    def get_action(self, state, reward=None):
        return random.choice(state.valid_actions())

