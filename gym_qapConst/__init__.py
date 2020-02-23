from gym.envs.registration import register

register(
    id='qapConst-v0',
    entry_point='gym_qapConst.envs:QapConstEnv',
)