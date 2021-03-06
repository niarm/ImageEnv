import numpy as np

class SimplePerceptionField():

    def __init__(self, id, startPosition = (0,0), shape=(100,100)):
        self.id = id
        self.shape = shape
        self.startPosition = startPosition
        self.environmentSize = (500,500)
        self.actionSpace = np.zeros(5)

        self.pos = np.asarray(self.startPosition)
        self.vel = np.zeros(2)
        self.acc = np.zeros(2)
        self.maxspeed = 4;

        self.painterPos = np.asarray( (self.shape[0] * 0.5, self.shape[1]*0.5 )).astype(int)
        self.painterVel = np.zeros(2)
        self.painterAcc = np.random.rand(2)
        self.shouldPaint = False
        self.painterMaxspeed = 8
        

    @property
    def boundingBox(self):
        return np.asarray([ self.pos[0], self.pos[1], self.pos[0]+self.shape[0], self.pos[1]+self.shape[1] ]).astype(int)

    @property
    def painterPosition(self):
        return self.painterPos.astype(int)

    def setEnvironmentSize(self, size):
        self.environmentSize = (size[0], size[1])
        print("got new env-size",self.environmentSize)

    def step(self, action):
        '''Takes an action and performs an update on the Window and the painter
        
        Arguments:
            action {list} -- ['acc_x', 'acc_y', 'painter_acc_x','painter_acc_y', 'paint']
        '''
        #print("action:", action)
        assert action.shape == self.actionSpace.shape

        self.acc = np.asarray( (action[0], action[1] ) )
        self.painterAcc = np.asarray( (action[2], action[3] ) )
        self.shouldPaint = action[4]

        self.update()


    def update(self):
        self.vel = np.add(self.vel, self.acc)
        self.vel = np.clip(self.vel, -self.maxspeed, self.maxspeed)
        self.pos = np.add(self.pos, self.vel)
        self.acc = np.multiply(self.acc, 0.0)

        self.painterVel = np.add(self.painterVel, self.painterAcc)
        self.painterVel = np.clip(self.painterVel, -self.painterMaxspeed, self.painterMaxspeed)
        self.painterPos = np.add(self.painterPos, self.painterVel)
        self.acc = np.multiply(self.acc, 0.0)

        #print("painterPos[0]", self.painterPos[0])
        #print("painterPos[1]", self.painterPos[1])
        #print("self.shape[0]", self.shape[0])
        #print("self.shape[1]", self.shape[1])

        ### BOUNDING BOX OF MAIN IMAGE WITH SLIDING IMAGE
        if self.pos[0] <= 0:
            self.hardStop()
            self.pos[0] = 0

        if self.pos[0]+self.shape[0] >= self.environmentSize[1]:
            self.hardStop()
            self.pos[0] = self.environmentSize[1]

        if self.pos[1] <= 0:
            self.hardStop()  
            self.pos[1] = 0

        if self.pos[1]+self.shape[1] >= self.environmentSize[0]:
            self.hardStop()
            self.pos[1] = self.environmentSize[0]


        ### BOUNDING BOX OF PAINTER WITH SLIDING WINDOW
        if self.painterPos[0] <= 0:
            self.painterPos[0] = 0
            self.hardStopPainter()
        
        if self.painterPos[0] >= self.shape[0]:
            self.painterPos[0] = self.shape[0]
            self.hardStopPainter()

        if self.painterPos[1] <= 0:
            self.painterPos[1] = 0
            self.hardStopPainter()
        
        if self.painterPos[1] >= self.shape[1]:
            self.painterPos[1] = self.shape[1]
            self.hardStopPainter()
        
        #print("pos: {} , vel: {} , acc: {}".format(self.pos, self.vel, self.acc))

    def applyForce(self, forceVector):
        self.acc = np.add(self.acc, forceVector)

    def hardStop(self):
        self.vel = np.zeros(2)
        self.acc = np.zeros(2)

    def hardStopPainter(self):
        self.painterVel = np.zeros(2)
        self.painterAcc = np.zeros(2)

    def reset(self):
        #self.hardStop()
        #self.pos = np.asarray(self.startPosition)
        pass