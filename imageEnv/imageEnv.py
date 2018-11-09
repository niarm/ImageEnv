import gym
from gym import error, spaces, utils
from gym.utils import seeding
from .perceptionFields import SimplePerceptionField

import cv2
import numpy as np
from matplotlib.image import imread

class ImageEnv(gym.Env):
  metadata = {'render.modes': ['human']}

  def __init__(self, mainImageDimensions = (800,800), mapImageDimension=(100,100), perceptionFieldSize = (100,100)):
    self.imagePaths = []
    self.maskPaths = []

    self.mainImageDimensions = np.asarray(mainImageDimensions)
    self.mapImageDimension = np.asarray(mapImageDimension)
    self.perceptionFieldSize = np.asarray(perceptionFieldSize)

    self.images = []
    self.masks = []

    self.currentImage = None
    self.currentImageID = None
    self.currentMask = None
    self.renderedOutput = np.zeros_like(self.mainImageDimensions)

    self.perceptionFields = []
    self.createPerceptionField(id=1)


  def createPerceptionField(self, id):
    newPF = SimplePerceptionField(id, startPosition=(0,0), shape=self.perceptionFieldSize)
    self.perceptionFields.append(newPF)

  def registerImagesAndMasks(self, imagePaths, maskPaths):
    self.imagePaths = imagePaths
    self.maskPaths = maskPaths
    
    if self.currentImage == None:
      self.loadNextImageAndMask(0)

  def loadNextImageAndMask(self, id):
    image = imread(self.imagePaths[id])
    mask = imread(self.maskPaths[id])

    self.images.append(image)
    self.masks.append(mask)

  def nextImage(self):
    if self.currentImageID == None:
      self.currentImageID = 0
    else:
      self.currentImageID += 1
    
    self.loadNextImageAndMask(self.currentImageID)

    self.currentImage = self.images[self.currentImageID]
    self.currentMask = self.masks[self.currentImageID]


  def step(self, action):
    pass

  def reset(self):
    #reset PerceptionFields
    for pf in self.perceptionFields:
      pf.reset()

    #Load next Image
    self.nextImage()
    

  def render(self, mode='human', close=False):
    cv2.imshow("ImageEnvironment", self.currentImage);