import numpy as np
import tensorflow as tf 
import tensorflow_hub as tfhub
import sentencepiece
import tf_sentencepiece

class USEFeatureExtractor():
    def __init__(self):
        self.model = None

    def transform(self, sentence:str):
        return np.zeros(300)