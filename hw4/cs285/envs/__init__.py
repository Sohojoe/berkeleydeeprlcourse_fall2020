from gym.envs.registration import register

def register_envs():
    register(
        id='marathon-hopper-v0',
        entry_point='cs285.envs.marathon_envs:HopperEnv',
        max_episode_steps=1000,
    )
    register(
        id='marathon-walker-v0',
        entry_point='cs285.envs.marathon_envs:WalkerEnv',
        max_episode_steps=1000,
    )
    register(
        id='marathon-ant-v0',
        entry_point='cs285.envs.marathon_envs:AntEnv',
        max_episode_steps=1000,
    )
