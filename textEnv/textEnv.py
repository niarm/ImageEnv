import gym
from gym import error, spaces, utils
from gym.utils import seeding

import numpy as np
import itertools
import random

from .textPerceptionField import TextPerceptionField
from .use_feature_extractor import USEFeatureExtractor

class TextSimilarityEnv(gym.Env):
  metadata = {'render.modes': ['human']}

  def __init__(self, n_compared_texts:int=2, feature_extractor=None):
    self.texts = []
    self.possible_combinations=None

    self.n_compared_texts = n_compared_texts
    self.current_text_ids = None
    self.current_texts = None
    self.already_seen_combinations = set()
    self.perception_fields = []
    self.miniatures = []

    self.feature_extractor = feature_extractor

    self.state = None

  def create_perception_fields(self, n_perception_fields:int=1):
    self.perception_fields = []
    #create n perception fields for each document
    for idx in range(self.n_compared_texts):
      pf_dict = {'pf_idx':idx, 'perception_fields':[]}
      for _ in range(n_perception_fields):
        newPF = TextPerceptionField(idx, startPosition=(0,))
        pf_dict['perception_fields'].append(newPF)
      self.perception_fields.append(pf_dict)
    print(f"{self.__class__.__name__}:: Created {self.perception_fields} perception Fields")

  def set_texts(self, texts:list, flush_existing_texts:bool=False):
    if flush_existing_texts==True:
      self.texts=[]
    self.texts+=texts
    combinations = itertools.combinations([idx for idx,_ in enumerate(self.texts)], 2) 
    self.possible_combinations = [x for x in combinations]
    print(f"{self.__class__.__name__}:: ENV has {len(self.texts)} texts with possible combinations: {self.possible_combinations} ")

  def next_text_pair(self):
    if len(self.possible_combinations)==0:
      return False

    # set next text-pair
    self.current_text_ids = self.possible_combinations.pop()
    self.current_texts = [self.texts[idx] for idx in self.current_text_ids]

    #reset perception fields
    self.reset_perception_fields()

    #calculate "miniatures":
    self.calculate_miniatures()

    return True
  
  def step(self, action = None):
    self.state = {}
    perception_results = []

    for pf_dict in self.perception_fields:
      for pf in pf_dict['perception_fields']:
        #random movement for testing
        testval = 1
        acc_test = round(random.uniform(-testval, testval))
        if acc_test >=0:
          acc_left=0
          acc_right=1
        else:
          acc_left=1
          acc_right=0


        pf.step({'acc_left':acc_left,
                 'acc_right':acc_right,
                 'increase_window': 0,
                 'decrease_window': 0})

        box = pf.boundingBox
        window = pf.perception_window
        
        perception_results.append({'t_id':pf.id,'box':box, 'window':window})
    
    self.state['perception_fields'] = perception_results
    self.state['miniatures'] = self.miniatures

    #state.append(self.perceptionResults)
    return self.state

  def reset_perception_fields(self):
    #reset perception-fields
    for pf_dict in self.perception_fields:
      perception_fields = pf_dict['perception_fields']
      for pf in perception_fields:
        pf.set_text(self.texts[ int( pf_dict['pf_idx'] ) ])
        pf.reset()

  def calculate_miniatures(self):
    miniatures = []
    vectors = self.feature_extractor.transform(self.current_texts)
    for idx, miniature_vec in enumerate(vectors):
      min_dict = {'t_id':idx, 'vectors':miniature_vec}
      miniatures.append(min_dict)
    
    self.miniatures = miniatures


  def render(self, mode='human', close=False):        
    #print(f"WINDOW: {render_output}")
    #print(f"STATE: {self.state['perception_fields']}")
    pass