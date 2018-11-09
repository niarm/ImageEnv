import gym
import imageEnv
import cv2
from time import sleep

if __name__== "__main__":
    env = gym.make('ImageEnv-v0')
    
    images = []
    masks = []
    for i in range(45,80):
        img = "data/input/images/"+str(i)+".jpg"
        mask = "data/input/masks/"+str(i)+".jpg"
        images.append(img)
        masks.append(mask)

    
    env.registerImagesAndMasks(imagePaths=images, maskPaths=masks)
    env.nextImage()
    count = 0
    
    while True:
        env.step()
        env.render()
        print("step #", count)
        count += 1
        if count > 2000:
            print("NEXT IMAGE")
            count = 0
            env.nextImage()