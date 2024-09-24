'''
This program reproduces the simulation environment provided by 
the following paper:
- H. Bayerlein, P. De Kerret and D. Gesbert, "Trajectory Optimization 
  for Autonomous Flying Base Station via Reinforcement Learning," IEEE 
  19th International Workshop on Signal Processing Advances in Wireless 
  Communications (SPAWC), 2018, pp. 1-5.

This environment has a map of 15-by-15 cells. The UAV starts from the left
bottom and moves up/down/left/right to the next cell to maximize the
sum rate. Each cell is indexed by (col,row) as shown below:
```
            0   1   2   3   ... COLS-1
          +---+---+---+---+ ... +---+
        0 |   |   |   |   |     |   | 
          +---+---@---+---+ ... +---+   <-- @:UE1
        1 |   |   |   |   |     |   | 
          +---+---+---+---+ ... @---+   <-- @:UE2
        2 |   |   | O |   |     |   |   <-- O:UAV
          +---+---+---+---+ ... +---+
          .                         .
          .                         .
          +---+---+---+---+ ... +---+
   ROWS-1 |   |   |   |   |     |   |
          +---+---+---+---+ ... +---+
```
'''

from uavenv.base import UAVEnvBase

import pygame
import math
from   shapely.geometry import LineString, Polygon

class Frame:
    '''
    This class defines the parameters of the 2D layout for the map.
    Customizable layout parameters are:
    - ROWS, COLS: define how many horizontal and vertical cells in
      the map.
    - ALTITUDE: define the UAV flying altitude
    - INITIAL_METER_PER_PIXEL: pixel to meter conversion
    '''

    ## map setup
    FOOTER_SPACE = 80             # footer space for text messages
    WIDTH = 800                   # screen dimension (pixels) for the map
    HEIGHT = 800+FOOTER_SPACE     # screen dimension (pixels) for the map
    ROWS, COLS = 15, 15           # number of rows and columns
    ALTITUDE = 20                 # altitude of the UAV (in pixels)
    INITIAL_METER_PER_PIXEL = 2   # use it to convert to actual distance
                                  # - note: it's used for initial calculation,
                                  #       it's invalid after screen is resized

    CELL_WIDTH = WIDTH // COLS    # cell size (in pixels)
    CELL_HEIGHT = (HEIGHT-FOOTER_SPACE) // ROWS

    def resize_screen(width, height):
        '''It adjusts the screen size.'''
        Frame.WIDTH = width
        Frame.HEIGHT = height
        Frame.CELL_WIDTH = Frame.WIDTH // Frame.COLS
        Frame.CELL_HEIGHT = (Frame.HEIGHT-Frame.FOOTER_SPACE) // Frame.ROWS

    def object_size():
        '''It recommends the size of an object based on the size of a cell.'''
        return min(Frame.CELL_WIDTH//3,Frame.CELL_HEIGHT//3)

    def cell_center_xy(pos):
        '''It returns the center (x,y) screen coordinate of the cell.'''
        x = pos.col*Frame.CELL_WIDTH + Frame.CELL_WIDTH//2
        y = pos.row*Frame.CELL_HEIGHT + Frame.CELL_HEIGHT//2
        return (x,y)

    def cell_distance(p1, p2, ground_to_air=False):
        '''It returns the distance (in pixels) between two coordinates.'''
        x1,y1 = Frame.cell_center_xy(p1)
        x2,y2 = Frame.cell_center_xy(p2)
        altitude = Frame.ALTITUDE if ground_to_air else 0
        return math.sqrt(altitude**2 + (x1-x2)**2 + (y1-y2)**2)
    

class Pos:
    '''
    This is an internal class used for managing coordination.
    '''
    def __init__(self, col, row):
        self.col = col
        self.row = row
    def __eq__(self, other):
        return self.col==other.col and self.row==other.row
    def copy(self):
        return Pos(self.col,self.row)
    
############################################
## define world objects for rendering
############################################

class CELL:
    '''
    Define the skin property of a cell.
    '''
    BACKGROUND = (255, 255, 255)  # white color
    START_COLOR = (0, 100, 0)     # dark green
    END_PEN_COLOR = (0, 0, 0)     # pen color to draw a cross in the final cell
    GRID_COLOR = (0, 0, 0)        # black color for grid lines

class OBSTACLE:
    '''
    Define skin property of an obstacle and the following:
    - POS_LIST: a list containing the locations of all obstacles. Make
      sure that they are not outside of the 2D map.
    '''
    POS_LIST = []                 # list of positions of the cells
    COLOR = (70, 70, 70)          # dark grey
    for x in range(9,11):
        for y in range(8,12):
            POS_LIST.append(Pos(x,y))

class UAV:
    '''
    Define skin property of the UAV and the following:
    - START_POS: the start position
    - END_POS: the returning position
    - FLIGHT_TIME: the time step that the UAV can fly
    '''
    COLOR = (255, 0, 0)               # red color for the UAV
    START_POS = Pos(0, Frame.ROWS-1)  # left-bottom
    END_POS = Pos(0, Frame.ROWS-1)    # left-bottom
    FLIGHT_TIME = 50                  # flight time of the UAV including returning

class UE:
    '''
    Define skin property of a UE and the following:
    - POS: a dict collecting the locations of all UEs
    '''
    COLOR = (0, 0, 255)           # blue color for UEs
    POS = {0:Pos(4.5, 2.5),       # list of UEs, including their location
           1:Pos(11.5, 6.5)}      #   which is on the grid
    RATE = {}                     # to be filled with the rate per UE per cell

class SHADOW:
    '''
    Define skin property of shadows. The setup is automatically done.
    '''
    COLOR1 = (200, 200, 200)      # very light grey
    COLOR2 = (128, 128, 128)      # light grey for more than 1 blockage
    NLOS = [] # 3D array [ue_id][col][row] -> 0:NoBlockage, 1:Blocked
    BLOCKAGE = [[0 for _ in range(Frame.ROWS)] for _ in range(Frame.COLS)] # blockage count
    ## detect blockage, i.e. non-line-of-sight
    for id,pos in UE.POS.items():
        NLOS.append([[0 for _ in range(Frame.ROWS)] for _ in range(Frame.COLS)])
        p1 = Frame.cell_center_xy(pos)
        for col in range(Frame.COLS):
            for row in range(Frame.ROWS):
                NLOS[id][col][row] = 0   # no blockage, line-of-sight (LOS)
                p2 = Frame.cell_center_xy(Pos(col,row)) # center position of cell (col,row)
                line1 = LineString([p1, p2])
                for obstacle in OBSTACLE.POS_LIST:
                    x,y = obstacle.col*Frame.CELL_WIDTH, obstacle.row*Frame.CELL_HEIGHT
                    rect1 = Polygon([(x, y), (x+Frame.CELL_WIDTH, y), 
                                     (x+Frame.CELL_WIDTH, y+Frame.CELL_HEIGHT), 
                                     (x, y+Frame.CELL_HEIGHT)])
                    if line1.intersects(rect1):
                        NLOS[id][col][row] = 1 # blocked, set 1 to Non line-of-sight
                        break
    ## consolidate blockage from all UEs into `BLOCKAGE[col][row]`
    for id in UE.POS:
        for col in range(Frame.COLS):
            for row in range(Frame.ROWS):
                BLOCKAGE[col][row] += NLOS[id][col][row]

#############################################
## communication model
#############################################

class COMM:
    '''
    Define the communication model. Customizable parameters are
    provided as constants in the class.
    '''

    FREQ  = 2.4e9  # operating frequency in Hz (2.4GHz)
    ALPHA = 2      # pathloss exponent
    SIGMA = 1      # Rayleigh fading scaling factor
    BETA_LOS  = 1     # shadowing attenuation for LOS
    BETA_NLOS = 0.01  # shadowing attenuation for NLOS

    NOISE_dBm = -174  # dBm, this is the thermal noise per Hz
    NOISE = (10**(NOISE_dBm/10)) / 1000       # in linear scale

    TX_POWER_dBm = 15     # dBm, transmit power
    TX_POWER = (10**(TX_POWER_dBm/10)) / 1000 # in linear scale

    def get_rate(d, NLOS):
        beta = COMM.BETA_NLOS if NLOS else COMM.BETA_LOS
        pl = math.pow(d,-COMM.ALPHA) * beta       # pathloss
        sn_ratio = COMM.TX_POWER*pl / COMM.NOISE  # signal-to-noise ratio, linear scale
        return math.log2(1+sn_ratio)              # Shannon's rate bps per Hz

    def get_ue_rate_matrix():
        ue_rate = {}
        for ue_id,ue_pos in UE.POS.items():
            ue_rate[ue_id] = [[0 for _ in range(Frame.ROWS)] 
                              for _ in range(Frame.COLS)]    # create a 2D array with zeroes
            for col in range(Frame.COLS):
                for row in range(Frame.ROWS):
                    d = Frame.cell_distance(ue_pos,Pos(col,row),ground_to_air=True) \
                        * Frame.INITIAL_METER_PER_PIXEL
                    ue_rate[ue_id][col][row] = COMM.get_rate(d, SHADOW.NLOS[ue_id][col][row])
        return ue_rate

## fill UE.RATE matrix, use this one time calculation 
## to avoid repeating the same calculation over and over again,
UE.RATE = COMM.get_ue_rate_matrix()

############################################
## define state & action classes
############################################

class State:
    '''
    This is the system state class observing the UAV's (x,y) and
    flight time.
    '''

    valid_action_matrix = None # to be provided

    def __init__(self, col, row, step=0):
        ## state = (x,y,t)
        self.col = col   # x
        self.row = row   # y
        self.step = step # t

    def valid_actions(self):
        '''It returns a list of valid actions.'''
        return State.valid_action_matrix[self.col][self.row]
    
    def __str__(self):
        '''It returns a str representation of the state. It can be useful
        for indexing a state and showing meaningful states for debugging.'''
        return f"({self.col},{self.row},{self.step})"


class Action:
    '''
    This is the Action class. There are four valid actions: `UP`, 
    `DOWN`, `LEFT`, and `RIGHT`. The action uses integer as its native 
    data type.
    '''
    UP    = 0  # define all actions here starting from 0
    DOWN  = 1
    LEFT  = 2
    RIGHT = 3

    def __init__(self, val):
        self._val = val 

    def __int__(self):
        return self._val
    
    def __eq__(self, other):
        return self._val==int(other)


############################################
## define the environment
############################################

class UAVEnv(UAVEnvBase):
    '''
    This is the environment class.
    '''

    action_space = [Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT]

    def __init__(self):

        ## call base class
        super().__init__()
        self.message = "Use scroll wheel to change FPS, " \
                       "and left click to pause/resume."

        ## specify the environment info
        map_width = Frame.WIDTH * Frame.INITIAL_METER_PER_PIXEL
        map_height = (Frame.HEIGHT-Frame.FOOTER_SPACE) * Frame.INITIAL_METER_PER_PIXEL
        uav_altitude = Frame.ALTITUDE * Frame.INITIAL_METER_PER_PIXEL
        self.info = {}
        self.info["description"] = {
            "area": f"The area is {map_width}-by-{map_height} in meters",
            "altitude": f"UAV is flying at {uav_altitude} meters above ground",
            "height": f"The flying altitude is about the height of a {uav_altitude*0.3:.0f}-story building"
        }
        self.info["flight_time"] = UAV.FLIGHT_TIME

        ## defines valid movements in the state class
        valid_actions = [[[Action.UP,Action.DOWN,Action.LEFT,Action.RIGHT] 
                                        for _ in range(Frame.ROWS)] 
                                        for _ in range(Frame.COLS)]
        for col in range(Frame.COLS):
            valid_actions[col][0].remove(Action.UP) # can't move up at top row
            valid_actions[col][Frame.ROWS-1].remove(Action.DOWN) # can't move down at bottom row
        for row in range(Frame.ROWS):
            valid_actions[0][row].remove(Action.LEFT) # can't move left at leftmost col
            valid_actions[Frame.COLS-1][row].remove(Action.RIGHT) # can't move right at rightmost col
        for pos in OBSTACLE.POS_LIST:
            valid_actions[pos.col-1][pos.row].remove(Action.RIGHT) # can't enter the obstacle
            valid_actions[pos.col+1][pos.row].remove(Action.LEFT)  # around it from any 
            valid_actions[pos.col][pos.row-1].remove(Action.DOWN)  # directions
            valid_actions[pos.col][pos.row+1].remove(Action.UP)
        State.valid_action_matrix = valid_actions

        ## initialize pygame
        pygame.init()
        pygame.display.set_caption("UAV Trajectory Optimization")
        self.font = pygame.font.SysFont(None, 55)   # label font
        self.font2 = pygame.font.SysFont(None, 36)  # message font
        self.font3 = pygame.font.SysFont(None, 24)  # tiny font
        self.clock = pygame.time.Clock()
        self.is_first_render = True

        ## initialize environment
        self.reset()

    def reset(self):
        '''It resets the environment to start a new episode. An
        episode may end if either it reaches the end of the 
        episode (i.e. terminated) or it encounters a stopping 
        condition (i.e. truncated).'''
        self.uav_pos = UAV.START_POS.copy()
        self.uav_step = 0
        return State(self.uav_pos.col, self.uav_pos.row, self.uav_step), \
               self.info

    def step(self, action):
        '''It executes the given action to the environment and 
        returns the new state, reward, conditions of termination
        and truncation, and environment information.'''

        ## apply the given action
        if action==Action.UP and self.uav_pos.row>0:
            self.uav_pos.row -= 1
        elif action==Action.DOWN and self.uav_pos.row<Frame.ROWS-1:
            self.uav_pos.row += 1
        elif action==Action.LEFT and self.uav_pos.col>0:
            self.uav_pos.col -= 1
        elif action==Action.RIGHT and self.uav_pos.col<Frame.COLS-1:
            self.uav_pos.col += 1

        ## move forward one time step
        self.uav_step += 1

        ## calculate reward
        all_rates = []
        for ue_id in UE.RATE:
            all_rates.append(UE.RATE[ue_id][self.uav_pos.col][self.uav_pos.row])
        reward = min(all_rates) 

        ## check for termination 
        ## i.e. UAV returned at the predefined flight time
        terminated = False
        truncated = False
        if self.uav_pos==UAV.END_POS and self.uav_step==UAV.FLIGHT_TIME:
            terminated = True

        ## check for truncation, case 1: UAV returns too early?
        elif self.uav_pos==UAV.END_POS:
            truncated = True
            reward -= reward * (UAV.FLIGHT_TIME - self.uav_step) # penalty

        ## check for truncation, case 2: UAV failed to return?
        elif self.uav_step==UAV.FLIGHT_TIME:
            truncated = True
            reward -= reward * 10 # penalty
        
        return State(self.uav_pos.col, self.uav_pos.row, self.uav_step), \
               reward, terminated, truncated, self.info

    def show_three_floats(self, col, row, values):
        '''Show up to three floats on this cell, for debugging purpose.'''
        margin = 3
        if values[0] is not None:
            img = self.font3.render(f"{values[0]:.1f}", True, CELL.GRID_COLOR)
            x = (Frame.CELL_WIDTH - img.get_rect().width) / 2
            y = margin
            self.screen.blit(img, (col*Frame.CELL_WIDTH+x, row*Frame.CELL_HEIGHT+y))
        if values[1] is not None:
            img = self.font3.render(f"{values[1]:.1f}", True, CELL.GRID_COLOR)
            x = (Frame.CELL_WIDTH - img.get_rect().width) / 2
            y = (Frame.CELL_HEIGHT - img.get_rect().height) / 2
            self.screen.blit(img, (col*Frame.CELL_WIDTH+x, row*Frame.CELL_HEIGHT+y))
        if values[2] is not None:
            img = self.font3.render(f"{values[2]:.1f}", True, CELL.GRID_COLOR)
            x = (Frame.CELL_WIDTH - img.get_rect().width) / 2
            y = Frame.CELL_HEIGHT - img.get_rect().height - margin
            self.screen.blit(img, (col*Frame.CELL_WIDTH+x, row*Frame.CELL_HEIGHT+y))

    def render(self, fps=60):
        '''It render the objects onto the screen. This is called when
        animation is requested.'''

        ## set mode before the first rendering
        if self.is_first_render:
            self.screen = pygame.display.set_mode((Frame.WIDTH, Frame.HEIGHT), pygame.RESIZABLE)
            self.is_first_render = False

        ## check fps setting
        if fps<1: fps=1
        if fps>120: fps = 120

        ## background
        self.screen.fill(CELL.BACKGROUND)

        ## set grid properties for drawing
        grid = []
        ## - background
        for row in range(Frame.ROWS):
            for col in range(Frame.COLS):
                grid.append((row,col,CELL.BACKGROUND))
        ## - shadow
        for row in range(Frame.ROWS): 
            for col in range(Frame.COLS):
                if SHADOW.BLOCKAGE[col][row]==1:
                    grid.append((row,col,SHADOW.COLOR1))
                elif SHADOW.BLOCKAGE[col][row]>1:
                    grid.append((row,col,SHADOW.COLOR2))
        ## - obstacle cells
        for pos in OBSTACLE.POS_LIST:
            grid.append((pos.row,pos.col,OBSTACLE.COLOR))
        ## - start cell
        grid.append((UAV.START_POS.row,UAV.START_POS.col,CELL.START_COLOR))

        ## draw grid
        for (row,col,color) in grid:
            rect = pygame.Rect(col*Frame.CELL_WIDTH, row*Frame.CELL_HEIGHT, 
                               Frame.CELL_WIDTH, Frame.CELL_HEIGHT)
            pygame.draw.rect(self.screen, color, rect)
            pygame.draw.rect(self.screen, CELL.GRID_COLOR, rect, 1)

        ## mark the final cell with a cross
        cross_left   = UAV.END_POS.col*Frame.CELL_WIDTH+5
        cross_right  = (UAV.END_POS.col+1)*Frame.CELL_WIDTH-5
        cross_top    = UAV.END_POS.row*Frame.CELL_HEIGHT+5
        cross_bottom = (UAV.END_POS.row+1)*Frame.CELL_HEIGHT-5
        pygame.draw.line(self.screen, CELL.END_PEN_COLOR,
                         (cross_left,cross_top), (cross_right,cross_bottom), 8)
        pygame.draw.line(self.screen, CELL.END_PEN_COLOR,
                         (cross_right,cross_top), (cross_left,cross_bottom), 8)

        ## draw UEs
        for pos in UE.POS.values():
            (x,y) = Frame.cell_center_xy(Pos(pos.col,pos.row))
            pygame.draw.circle(self.screen, UE.COLOR, (x,y), Frame.object_size())

        ## draw UAV
        (x,y) = Frame.cell_center_xy(Pos(self.uav_pos.col,self.uav_pos.row))
        pygame.draw.circle(self.screen, UAV.COLOR, (x,y), Frame.object_size())

        ## show message
        img = self.font2.render(self.message, True, CELL.GRID_COLOR)
        self.screen.blit(img, (10, Frame.ROWS*Frame.CELL_HEIGHT+5))

        ## show status
        status = f"FPS = {fps}  |  Step = {self.uav_step}"
        img = self.font2.render(status, True, CELL.GRID_COLOR)
        self.screen.blit(img, (10,Frame.HEIGHT-img.get_rect().height))

        ## render the world
        pygame.display.flip()
        self.clock.tick(fps if fps<120 else 0)

        ## check if the screen has been resized
        pygame.event.pump()
        if pygame.event.peek(pygame.VIDEORESIZE):
            events = pygame.event.get()
            for event in events:
                if event.type==pygame.VIDEORESIZE:
                    Frame.resize_screen(event.w, event.h)

