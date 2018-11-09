import gym
from gym import error, spaces, utils
from gym.utils import seeding
from .perceptionFields import SimplePerceptionField

import cv2
import numpy as np
from PIL import Image
import random

class ImageEnv(gym.Env):
  metadata = {'render.modes': ['human']}

  def __init__(self, mapImageDimension=(100,100), perceptionFieldSize = (100,100)):
    self.imagePaths = []
    self.maskPaths = []

    self.mainImageDimension = None
    self.mapImageDimension = np.asarray(mapImageDimension)
    self.perceptionFieldSize = np.asarray(perceptionFieldSize)

    self.images = []
    self.masks = []
    self.minimaps = []
    self.perceptionResults = []
    
    self.currentImage = None
    self.currentImageID = None
    self.currentMask = None
    self.currentMiniMap = None

    self.renderedOutput = np.zeros_like(self.mainImageDimension)

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
    image = cv2.imread(self.imagePaths[id])
    mask = cv2.imread(self.maskPaths[id])
    minimap  = cv2.resize(image, dsize=(128, 140), interpolation=cv2.INTER_CUBIC)

    self.images.append(image)
    self.masks.append(mask)
    self.minimaps.append(minimap)

  def nextImage(self):
    if self.currentImageID == None:
      self.currentImageID = 0
    else:
      self.currentImageID += 1
    
    self.loadNextImageAndMask(self.currentImageID)

    self.currentImage = self.images[self.currentImageID]
    self.currentMask = self.masks[self.currentImageID]
    self.currentMiniMap = self.minimaps[self.currentImageID]

    self.mainImageDimension = self.currentImage.shape
    
    for pf in self.perceptionFields:
      pf.reset()
      pf.setEnvironmentSize(self.mainImageDimension)

  def step(self, action=None):
    state = []

    self.perceptionResults = []
    for pf in self.perceptionFields:
      testval = 4.0
      pf.step(np.asarray([ random.uniform(-testval, testval), random.uniform(-testval, testval), random.uniform(-testval, testval), random.uniform(-testval, testval), 0 ]))
      box = pf.boundingBox
      perceptionVisibleWindow = self.currentImage[box[1]:box[3], box[0]:box[2], :]
      self.perceptionResults.append(perceptionVisibleWindow)

    state.append(self.perceptionResults)
    return state

  def reset(self):
    #reset PerceptionFields
    for pf in self.perceptionFields:
      pf.reset()

    #Load next Image
    self.nextImage()


  def render(self, mode='human', close=False):
    #create empty image
    self.renderedOutput = np.zeros(self.mainImageDimension).astype('uint8')
    
    #draw main image
    self.renderedOutput += self.currentImage

    # draw mask
    transposedMask = np.transpose(self.currentMask, (2,0,1))
    zeroMask = np.zeros_like(transposedMask)
    zeroMask[1] = transposedMask[0] 
    drawMask = np.transpose(zeroMask, (1,2,0))
    
    #self.renderedOutput = cv2.add(self.renderedOutput,drawMask)
    self.renderedOutput = cv2.addWeighted(self.renderedOutput,1.0, drawMask,0.3,0)

    #draw PerceptionFields
    for indx, pf in enumerate(self.perceptionFields):
      box = pf.boundingBox
      painter = pf.painterPosition

      cv2.rectangle(self.renderedOutput, (box[0], box[1]), (box[2], box[3]) ,(255,0,0), 1 )
      cv2.circle(self.renderedOutput,(box[0]+painter[0], box[1]+painter[1]), 5, (0,0,255), thickness=1)
      #print("PR: {} / {} -> {}".format(indx, len(self.perceptionResults), self.perceptionResults[indx]))
      cv2.imshow("PerceptionResult"+str(indx), self.perceptionResults[indx]);

    cv2.imshow("ImageEnvironment::MINIMAP", self.currentMiniMap);
    cv2.imshow("ImageEnvironment::MAIN", self.renderedOutput);
    cv2.waitKey(40)