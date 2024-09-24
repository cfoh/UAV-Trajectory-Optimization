from abc import abstractmethod

class RL:
    '''
    This class provides a base class for reinforcement learning mode. 
    It is an abstract class, subclass must implement the `get_action()`
    method.

    The constructor takes the following inputs parameters.

    Parameters:
    name : str
        The name of the model.
    '''

    def __init__(self, name="Base Class"):
        ## declare all visible properties
        self.name = name
        self.is_exploration = None

    @abstractmethod
    def get_action(self, state, reward=None):
        pass

    def set_exploration(self, exploration:bool):
        '''This method sets a new exploration flag (True/False) and returns
        the original setting.'''
        original_setting = self.is_exploration
        self.is_exploration = exploration
        return original_setting

    def load_data(self) -> int:
        '''This method loads data to the model.'''
        return -1 # failed or NA

    def save_data(self, round) -> bool:
        '''This method save data to a file.'''
        return False # failed or NA

## decaying float number for epsilon
class DecayingFloat:
    '''
    This class provides a delaying float number. It is disguised as a 
    `float` but provides methods to trigger a decay.

    The constructor takes the following inputs parameters.

    Parameters:
    value : float
        The initial value of the decaying float number. We assume it
        is a positive value.
    factor : float, optional, default=None
        The decaying factor. If None is specified, the float number will
        not decay.
    minval : float, optional, default=None
        The minimum value of the float. If None is specified, the float
        number can reach zero which is the lowest.
    mode : str
        It can be either "exp" for exponential decaying or "linear" for
        linear decaying. An unrecognized string will cause the value 
        not to decay.
    '''
    def __init__(self, value:float, factor:float=None, minval:float=None,
                 mode:str="exp"):
        self.init = value
        self.value = value
        self.factor = factor
        self.minval = minval
        self.mode = mode

    def __float__(self) -> float:
        '''
        This method performs the type casting operation to return a float.
        '''
        return float(self.value)

    def reset(self):
        '''
        To start over the decaying function from the beginning.
        '''
        self.value = self.init

    def decay(self):
        '''
        To perform a step of decay. The decaying depends on the `factor`
        and the `mode`. If `factor` is not given or `mode` string is
        unrecognized, the method simply does nothing.
        '''
        if self.factor==None: return

        if self.mode=="exp":      self.value *= self.factor
        elif self.mode=="linear": self.value -= self.factor
        
        if self.minval==None: 
            return
        elif self.value<self.minval:
            self.value = self.minval


