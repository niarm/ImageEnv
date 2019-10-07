import os
import numpy as np
import tensorflow as tf 
import tensorflow_hub as tfhub
import tf_sentencepiece

cache_path = "./tfhub_cache"
os.makedirs(cache_path)
os.environ["TFHUB_CACHE_DIR"] = cache_path 

class FeatureExtractor():
    def __init__(self, *args, **kwargs):
        pass
    def transform(self):
        pass

class USEFeatureExtractor(FeatureExtractor):
    MODULE_URL_MULTI = "https://tfhub.dev/google/universal-sentence-encoder-multilingual/1"
    MODULE_URL_EN = "https://tfhub.dev/google/universal-sentence-encoder/2"

    def __init__(self, language='de'):
        if language=='en':
            self.model_url = USEFeatureExtractor.MODULE_URL_EN 
        else:
            self.model_url = USEFeatureExtractor.MODULE_URL_MULTI
        
        self.model = tfhub.Module(self.model_url)
        self.session = tf.compat.v1.Session()
        self.session.run([tf.compat.v1.global_variables_initializer(), tf.compat.v1.tables_initializer()])

    def transform(self, sentences:list):
        print(f"transforming {len(sentences)} sentences" ,end="\r")
        vectors = self.session.run(self.model(sentences))
        return vectors