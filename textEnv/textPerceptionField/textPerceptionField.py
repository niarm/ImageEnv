import numpy as np
from dataclasses import dataclass
from textEnv.use_feature_extractor import FeatureExtractor

@dataclass
class TextPerceptionFieldAction:
    acc_left: bool = 0
    acc_right: bool = 0
    increase_window:bool = 0
    decrease_window:bool = 0
    extract_features:bool = 0

class TextPerceptionField():
    def __init__(self, id, startPosition = (0,), start_shape=(5,)):
        self.MIN_PERCEPTION_WINDOW_SIZE = 2
        self.MAX_PERCEPTION_WINDOW_SIZE = 20
        
        self.acc_speed = 1.0
        self.maxspeed = 3.0;

        self.id = id
        self.start_shape = np.asarray(start_shape)
        self.shape = self.start_shape
        self.startPosition = startPosition
        self.environmentSize = None
        
        # action-space= action {list} -- ['acc_left', 'acc_right' , 'increase_window', 'decrease_window']
        self.n_actions = 4
        self.last_action = None

        self.pos = np.asarray(self.startPosition)
        self.vel = np.zeros(1)
        self.acc = np.zeros(1)
        
        
        self.text = None
        #print(f"START: id: {self.id}, pos: {self.pos}, vel: {self.vel}, acc: {self.acc}, shape:{self.shape}")

    @property
    def boundingBox(self):
        #print(f"id: {self.id}, pos: {self.pos}")
        return np.asarray([ round(self.pos[0]), round(self.pos[0])+self.shape[0] ]).astype(int)
    
    @property
    def percentage_position_bounding_box(self):
        return np.divide(self.boundingBox, self.MAX_PERCEPTION_WINDOW_SIZE)

    @property
    def perception_window(self):
        text_list = self.text.split()
        window = " ".join(text_list[self.boundingBox[0] : self.boundingBox[1]])
        return window

    def set_text(self, text:str):
        self.text = text
        self.environmentSize = np.asarray( [len(self.text.split())] )
        self.MAX_PERCEPTION_WINDOW_SIZE = self.environmentSize[0]

    def step(self, action:TextPerceptionFieldAction):
        #set acceleration
        if action.acc_left==1:
            self.acc = np.asarray( [self.acc_speed]  )
        elif action.acc_right==1:
            self.acc = np.asarray( [-self.acc_speed]  )
        else:
            self.acc = np.zeros(1)
        
        #set perception-window size
        if (action.increase_window==1) and self.shape[0] < self.MAX_PERCEPTION_WINDOW_SIZE:
            self.shape = np.asarray([(self.shape[0]+1)])
        if (action.decrease_window==1) and self.shape[0] > self.MIN_PERCEPTION_WINDOW_SIZE:
            self.shape = np.asarray([(self.shape[0]-1)])
        
        self.last_action = action

        self.update()


    def update(self):
        self.vel = np.add(self.vel, self.acc)
        self.vel = np.clip(self.vel, -self.maxspeed, self.maxspeed)
        self.pos = np.add(self.pos, self.vel)
        #print(f"id: {self.id}, pos: {self.pos}, vel: {self.vel}, acc: {self.acc}, shape:{self.shape}")
        
        #reset acceleration
        self.acc = np.multiply(self.acc, 0.0)
        
        ### BOUNDING BOX OF MAIN TEXT WITH SLIDING PERCEPTION-FIELD_WINDOW
        if self.pos[0] <= 0:
            self.hardStop()
            self.pos[0] = 0

        if self.pos[0]+self.shape[0] >= self.environmentSize[0]:
            self.hardStop()
            self.pos[0] = self.environmentSize[0] - self.shape[0]


    def applyForce(self, forceVector):
        self.acc = np.add(self.acc, forceVector)

    def hardStop(self):
        self.vel = np.zeros(1)
        self.acc = np.zeros(1)

    def reset(self):
        self.hardStop()
        self.pos = np.asarray(self.startPosition)
        self.shape = self.start_shape
        if self.shape[0] > self.MAX_PERCEPTION_WINDOW_SIZE:
            self.shape = np.asarray([self.MAX_PERCEPTION_WINDOW_SIZE])

        #print(f"RESET: id: {self.id}, pos: {self.pos}, vel: {self.vel}, acc: {self.acc}, shape:{self.shape}")