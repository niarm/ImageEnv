import numpy as np

class SimplePerceptionField():

    def __init__(self, id, startPosition = (0,0), shape=(100,100)):
        self.id = id
        self.shape = shape
        self.startPosition = startPosition
        
        self.pos = np.asarray(self.startPosition)
        self.vel = np.zeros(2)
        self.acc = np.zeros(2)

        self.maxspeed = 2;
        self.maxforce = 0.05;

    @property
    def boundingBox(self):
        return np.asarray([ self.pos[0], self.pos[1], self.shape[0], self.shape[1] ])


    def update(self):
        print("update")
        self.vel = np.add(self.vel, self.acc)
        self.vel = np.clip(self.vel, -self.maxspeed, self.maxspeed)
        self.pos = np.add(self.pos, self.vel)

        self.acc = np.multiply(self.acc, 0.0)
        print("pos: {} , vel: {} , acc: {}".format(self.pos, self.vel, self.acc))

    def applyForce(self, forceVector):
        self.acc = np.add(self.acc, forceVector)

    def hardStop(self):
        self.vel = np.zeros(2)
        self.acc = np.zeros(2)

    def reset(self):
        self.hardStop()
        self.pos = np.asarray(self.startPosition)