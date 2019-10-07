import gym
import imageEnv
import cv2
from time import sleep

if __name__== "__main__":
    env = gym.make('ImageEnv-v0')
    
    images = []
    masks = []
    for i in range(1,80):
        img = "data/DAVIS/JPEGImages/Full-Resolution/bear/"+'{0:05d}'.format(i)+".jpg"
        mask = "data/DAVIS/Annotations/Full-Resolution/bear/"+'{0:05d}'.format(i)+".png"
        
        images.append(img)
        masks.append(mask)

    env.registerImagesAndMasks(imagePaths=images, maskPaths=masks, scaleFactor=0.5)
    env.nextImage()
    count = 0
    
    while True:
        env.step()
        #print("step #", count)

        count += 1
        env.render()

        if count > 1000:
            print("NEXT IMAGE")
            count = 0
            env.nextImage()