import numpy as np
from gym import utils

from mlagents_envs.environment import UnityEnvironment
from gym_unity.envs import UnityToGymWrapper
from mlagents_envs.side_channel.engine_configuration_channel import (
    EngineConfigurationChannel,
)
from sys import platform

from timeit import default_timer as timer
from datetime import timedelta
import os
import gym.spaces as spaces

def MarathonEnvs(
        environment_name: str,
        num_spawn_envs: int = 1,
        worker_id: int = 0,
        marathon_envs_path: str = None,
        no_graphics: bool = False,
        use_editor: bool = False,
        inference: bool = False,
    ):
    """
    Environment initialization
    :param environment_name: The Marathon Environment 
    :param num_spawn_envs: The number of environments to spawn per instance
    :param worker_id: Worker number for environment.
    :param marathon_envs_path: alternative path for environment
    :param no_graphics: Whether to run the Unity simulator in no-graphics mode
    :param use_editor: If True, assume Unity Editor is the envionment (use for debugging)
    :param inference: If True, run in inference mode (normal framerate)
    """
    use_visual: bool = False
    uint8_visual: bool = False
    multiagent: bool = True # force multiagent
    flatten_branched: bool = False
    allow_multiple_visual_obs: bool = False

    base_port = 5005
    # use if we want to work with Unity Editoe
    if use_editor:
        # base_port = DEFAULT_EDITOR_PORT
        marathon_envs_path = None
    elif marathon_envs_path is None:
        marathon_envs_path = os.path.join('envs', 'MarathonEnvs')
        if platform == "win32":
            marathon_envs_path = os.path.join(marathon_envs_path, 'Marathon Environments.exe')
    args = ['--spawn-env='+environment_name]
    args.append('--num-spawn-envs='+str(num_spawn_envs))

    engine_configuration_channel = EngineConfigurationChannel()
    channels = [engine_configuration_channel]

    unity_env = UnityEnvironment(
        marathon_envs_path,
        worker_id = worker_id,
        base_port=base_port,
        side_channels=channels,
        no_graphics=no_graphics,
        additional_args = args,
    )
    if not inference:
        engine_configuration_channel.set_configuration_parameters(
            width=160, height=160, quality_level=0, 
            time_scale=20., target_frame_rate=-1)
    return unity_env

class HopperEnv(UnityToGymWrapper, utils.EzPickle):
    def __init__(self):

        utils.EzPickle.__init__(**locals())
        env_name='Hopper-v0'
        num_envs = 1
        unity_env = MarathonEnvs(env_name, num_envs, inference=True)
        # unity_env = MarathonEnvs(env_name, num_envs)
        UnityToGymWrapper.__init__(self, unity_env)
        obs_space = spaces.Box(
            -1.,
            1.,
            (self.observation_space.shape),
            self.observation_space.dtype)
        self._observation_space = obs_space

    @property
    def spec(self):
        return self._spec
    
    @spec.setter
    def spec(self, spec):
        self._spec = spec

class WalkerEnv(UnityToGymWrapper, utils.EzPickle):
    def __init__(self):

        utils.EzPickle.__init__(**locals())
        env_name='Walker2d-v0'
        num_envs = 1
        unity_env = MarathonEnvs(env_name, num_envs, inference=True)
        UnityToGymWrapper.__init__(self, unity_env)
        obs_space = spaces.Box(
            -1.,
            1.,
            (self.observation_space.shape),
            self.observation_space.dtype)
        self._observation_space = obs_space

    @property
    def spec(self):
        return self._spec
    
    @spec.setter
    def spec(self, spec):
        self._spec = spec

class AntEnv(UnityToGymWrapper, utils.EzPickle):
    def __init__(self):

        utils.EzPickle.__init__(**locals())
        env_name='Ant-v0'
        num_envs = 1
        unity_env = MarathonEnvs(env_name, num_envs, inference=True)
        UnityToGymWrapper.__init__(self, unity_env)
        obs_space = spaces.Box(
            -1.,
            1.,
            (self.observation_space.shape),
            self.observation_space.dtype)
        self._observation_space = obs_space

    @property
    def spec(self):
        return self._spec
    
    @spec.setter
    def spec(self, spec):
        self._spec = spec