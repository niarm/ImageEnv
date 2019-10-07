from gym.envs.registration import registry, register, make, spec

register(
    id='TextSimilarityEnv-v0',
    entry_point='textEnv.textEnv:TextSimilarityEnv',
)

from .textPerceptionField import TextPerceptionField
from .use_feature_extractor import USEFeatureExtractor, FeatureExtractor