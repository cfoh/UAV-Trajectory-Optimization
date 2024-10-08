'''
The module contains the a base class for the UAV environment. 
It also defines the specifications of `Action` and `State` classes.
'''

import pygame
from abc import abstractmethod

class Action:
    '''
    This class defines the Action.
    '''
    pass

class State:
    '''
    This class defines the system state.
    '''

    def valid_actions(self):
        '''It returns a list or range of valid actions. 
        The returning data type should be clearly defined in the 
        actual environment implementation.'''
        return None

    def __str__(self):
        '''It returns a str representation of the state. It can 
        be useful for indexing a state and showing meaningful 
        states for debugging.'''
        return ""


class UAVEnvBase:
    '''
    This is the base class for the UAV environment. It defines the 
    following constants:
    - EVENT_QUIT: when a close event is triggered by users
    - EVENT_MOUSEBUTTONDOWN: when a mouse button down event is detected
    - MOUSE_LEFT_BUTTON: check if a mouse button down is the left button
    - MOUSE_MIDDLE_BUTTON: check if a mouse button down is the middle button
    - MOUSE_RIGHT_BUTTON: check if a mouse button down is the right button
    - MOUSE_SCROLL_UP: check if a mouse button down is the 
      scrolling up of mouse wheel
    - MOUSE_SCROLL_DOWN: check if a mouse button down is the 
      scrolling down of mouse wheel

    It also defines the following methods:
    - reset(): call it to reset the episode
    - step(action): call it to provide an action for the environment to
      perform the action and return a new state and associated reward
    - render(): call it to render the environment onto the screen
    - get_event(): call it to detect a user event
    - set_message(message): call it with a message to show a message when
      rendering the environment
    - close(): call it to properly terminate the environment and release 
      all resources.
    '''

    ## event constants
    EVENT_QUIT            = pygame.QUIT
    EVENT_MOUSEBUTTONDOWN = pygame.MOUSEBUTTONDOWN
    MOUSE_LEFT_BUTTON   = 1
    MOUSE_MIDDLE_BUTTON = 2
    MOUSE_RIGHT_BUTTON  = 3
    MOUSE_SCROLL_UP     = 4
    MOUSE_SCROLL_DOWN   = 5

    def __init__(self):
        self.info = {}  # it should contain the environment info
        self.message = ""

    @abstractmethod
    def reset(self):
        '''This method resets the environment and returns the 
        initial system state.'''
        return None

    @abstractmethod
    def step(self, action):
        '''This method supplies an action to the environment.
        After the environment has performed the given action, it
        returns a tuple (state, reward, terminated, truncated, info)
        providing the new state, the reward due to the given action,
        whether the episode is terminated as it reaches the end, 
        whether it encounters a predefined condition and truncated,
        and the static information related to the environment.'''
        pass

    @abstractmethod
    def render(self):
        '''Call this method to render the environment onto the screen.'''
        pass

    def get_event(self):
        '''This method returns a user input event captured during 
        rendering. The input can be a keyboard or mouse event.'''
        return pygame.event.get()
    
    def set_message(self, message):
        '''Use this method to set a message to show on the environment.'''
        self.message = message
    
    def close(self):
        '''Use this method to close the environment and clear resources.'''
        pygame.quit()

