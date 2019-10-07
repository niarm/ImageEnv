import numpy as np
import tensorflow as tf 
import tensorflow_hub as tfhub
import tf_sentencepiece



class USEFeatureExtractor():
    MODULE_URL_DE = "https://tfhub.dev/google/universal-sentence-encoder-multilingual/1"

    def __init__(self):

        self.model = tfhub.Module(USEFeatureExtractor.MODULE_URL_DE)
        self.session = tf.compat.v1.Session()
        self.session.run([tf.compat.v1.global_variables_initializer(), tf.compat.v1.tables_initializer()])

    def transform(self, sentences:list):
        vectors = self.session.run(self.model(sentences))
        return vectors