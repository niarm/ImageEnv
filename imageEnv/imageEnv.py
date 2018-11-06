import gym
from gym import error, spaces, utils
from gym.utils import seeding

class ImageEnv(gym.Env):
  metadata = {'render.modes': ['human']}
  print("building Image-Environment")
  pass