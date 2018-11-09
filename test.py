import gym
import imageEnv
import cv2
from time import sleep

if __name__== "__main__":
    env = gym.make('ImageEnv-v0')
    
    target_index = 2

    img = "data/input/images/"+str(target_index)+".jpg"
    mask = "data/input/masks/"+str(target_index)+".jpg"
    
    env.registerImagesAndMasks(imagePaths=[img], maskPaths=[mask])
    env.nextImage()

    while True:
        env.step()
        env.render()