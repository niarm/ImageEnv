from gym.envs.registration import registry, register, make, spec

register(
    id='ImageEnv-v0',
    entry_point='imageEnv.imageEnv:ImageEnv',
)