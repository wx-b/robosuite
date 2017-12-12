from MujocoManip import SawyerPushEnv, SawyerStackEnv

### TODO: we should either explicitly include gym as a dependency ###
###       or come up with a more elegant solution for this.       ###

REGISTERED_ENVS = {'SawyerPushEnv' : SawyerPushEnv,
                   'SawyerStackEnv' : SawyerStackEnv}

def make(env_name, *args, **kwargs):
    """
    Try to get the equivalent functionality of gym.make in a sloppy way.
    """
    if env_name not in REGISTERED_ENVS:
        raise Exception('Environment not found. Make sure it is a registered environment in make.py')
    return REGISTERED_ENVS[env_name](*args, **kwargs)