# Extracts episodes from RobotTeleop data collections into a pickle of lists of episodes,
# where each episode is a tuple with a model.xml string and a list of MjSimStates.

import numpy as np
import sys
from MujocoManip import *
import mujoco_py
import pickle
import os
import zipfile
import time

from MujocoManip.models import *

def extract_file(f):
    if f.endswith('.zip'):
        d = '/tmp/%d.%d' % (int(time.time()), int(hash(f))) 
        print("Extracting zip file onto %s" % d)
        zip_ref = zipfile.ZipFile(f, 'r')
        zip_ref.extractall(d)
        zip_ref.close() 
    else:
        print("Reading directory of numpy arrays")
        d = f
    print(d, os.listdir(d))
    l = sorted(x for x in os.listdir(d) if x.startswith('state_'))
    st = np.vstack([np.load(os.path.join(d, i))["states"] for i in l])
    with open(os.path.join(d, 'model.xml')) as ff:
        xmlstring = ff.read()
    return (xmlstring, st)

def extract_files(l):
    return list(map(extract_file, l))

if __name__ == '__main__':
    if not sys.argv[-1].endswith(".pickle"):
        print("The last argument must be the .pickle you want to save to.")
        sys.exit(1)
    data = extract_files(sys.argv[1:-1])
    print("Saving to %s" % sys.argv[-1])
    with open(sys.argv[-1], 'wb') as p:
        pickle.dump(data, p, -1)

    print("Now rendering because why not")

    for _, st in data:
        env = make("BaxterLiftEnv",
                   ignore_done=True,
                   use_camera_obs=False,
                   gripper_visualization=True,
                   reward_shaping=True)

        obs = env.reset()

        dof = env.dof
        print('action space', env.action_space)
        print('Obs: {}'.format(len(obs)))
        print('DOF: {}'.format(dof))
        env.render()


        r = int((st.shape[1]-1)/2)
        for i in range(st.shape[0]):
            ss = st[i,:]
            state = mujoco_py.MjSimState(time=ss[0], qpos=ss[1:27], qvel=ss[27:], act=None, udd_state={})
            env.sim.set_state(state)
            env.sim.forward()
            # obs, reward, done, info = env.step(action)
            env.render()
