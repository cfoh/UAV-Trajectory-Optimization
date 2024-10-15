'''
This module contains class implementing Reinforcement Learning 
algorithm using State-Action-Reward-State-Action (SARSA).

- Actions must be an integer starting from 0
- States must have string representation and must provide `valid_actions()`
'''

import numpy as np
import random
import json, os
from   time import strftime, localtime

from rl.rlbase import RL, DecayingFloat

class SARSA(RL):
    '''
    It implements SARSA. The constructor takes two inputs:
    - `num_actions` specifies action spaces by providing the number of 
      possible actions
    - `exploration` is a flag specifying whether the algorithm should run
      in exploration-exploitation mode (when set to True), or in full 
      exploitation mode with no further exploration (when set to False)
    '''

    def __init__(self, num_actions, exploration=True):

        super().__init__("SARSA")
        self.is_exploration = exploration
        self.num_actions = num_actions

        ## Q-table
        self.q_table = {}

        ## ML hyperparameter
        self.alpha: float = 0.3    # learning rate
        self.gamma: float = 0.9    # discount factor
        #self.epsilon: float = 0.1  # exploration weight, 0=no; 1=full
        self.epsilon = DecayingFloat(value=0.9,factor=1.0-1e-8,minval=0.05)

        ## properties
        self.current_state = None
        self.current_action:int  = None

    def load_data(self) -> int:
        '''Load Q-table from `{ai.name}-load.json`.'''
        filename = f"{self.name}-load.json"
        if os.path.exists(filename):
            with open(filename, "r") as fp:
                self.q_table = json.load(fp)
            for state in self.q_table:
                if state=="round": continue
                if state=="epsilon":
                    if isinstance(self.epsilon,DecayingFloat):
                        self.epsilon.value = self.q_table[state]
                    else:
                        self.epsilon = self.q_table[state]
                    continue
                self.q_table[state] = np.array(self.q_table[state])
            if len(self.q_table)>1:
                print(f"- loaded '{filename}' containing {len(self.q_table)-1} states\n"
                      f"- stopped at round {self.q_table['round']}\n"
                      f"- epsilon is {float(self.epsilon)}")
            return self.q_table['round']
        else:
            print(f"- '{filename}' not found, no experience is used")
            return -1

    def save_data(self, round_id) -> bool:
        '''Save Q-table to `{ai.name}-[{date}][{time}].json`.'''
        class NpEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                else:
                    return super(NpEncoder, self).default(obj)

        ## write Q-Table to the json file
        ## this way, we don't lose the training data
        self.q_table["round"] = round_id
        self.q_table["epsilon"] = float(self.epsilon)
        filename = strftime(f"{self.name}-[%Y-%m-%d][%Hh%Mm%Ss].json", localtime())
        with open(filename, "w") as fp:
            json.dump(self.q_table, fp, cls=NpEncoder, indent=4)

        return True

    def q(self, state):
        '''It provides easy access to Q-table, i.e. use `q(s)[a]` to access
        the Q-value of state `s` and action `a`.'''
        if state not in self.q_table:
            ## create a row for this new state in Q-table
            self.q_table[state] = np.zeros(self.num_actions)
        return self.q_table[state]

    def get_action(self, state, reward=None) -> int:
        '''
        It returns an action based on SARSA.
        If `reward` is not provided, it will not update the q-table.
        SARSA: [current_state]-[current_action]-[reward]-[state]-[a1]
        - `reward` is based on `current_state` and `current_action`
        - `state` is the next state after transition
        - `a1` is the action to decide
        SARSA updates Q[current_state][current_action] based on the future
        reward when `a1` is executed.
        '''

        ## setup system info
        ## - s: current state (in str)
        ## - a: current action (in int)
        ## - reward: the reward after taking the action
        ## - s1: the state after taking the action (in str)
        ## - a1: the next action to take (in int)
        if self.current_state is not None: # skip first time
            s = str(self.current_state) # current state (before action)
            a = self.current_action     # the executed action
        s1 = str(state)        # next state (after action) from the environment 
        a1 = None              # to be decided

        ## choose the next action based on exploration-exploitation
        if self.is_exploration and random.uniform(0,1)<float(self.epsilon):
            ## do exploration
            a1 = random.choice(state.valid_actions())
        else:
            ## do exploitation
            ## may have multiple same max value
            ## ie. pi_star(s) = argmax_a(Q_star(s,a))
            max_value = np.max(self.q(s1)[state.valid_actions()]) # find the max value
                                                                  # only for valid actions
            possible_actions = []
            for i in state.valid_actions():
                if self.q(s1)[i]==max_value:
                    possible_actions.append(i) # add all carrying max value
            a1 = random.choice(possible_actions)

        if isinstance(self.epsilon, DecayingFloat):
            self.epsilon.decay()

        ## update Q-table using using SARSA based on the received reward:
        ## Q_next(s,a) = Q(s,a) \
        ##               + alpha * (reward + gamma*Q(s1,a1) - Q(s,a))
        ##             = (1-alpha) * Q(s,a)
        ##               + alpha  * (reward + gamma*Q(s1,a1))
        ## skip update if no reward is given or it's run for the very first time
        if self.current_state is not None and reward is not None:
            self.q(s)[a] = self.q(s)[a] \
                    + self.alpha * (reward + self.gamma*self.q(s1)[a1] - self.q(s)[a])

        ## return the action, also save (state,action) pair
        self.current_state = state
        self.current_action = a1
        return a1


