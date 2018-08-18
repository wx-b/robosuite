import time
from RoboticsSuite import make
import numpy as np
from RoboticsSuite.environments.demo_sampler import DemoSampler


class demo_config:
    use_demo = False
    adaptive = False
    num_traj = 1
    # params for open loop reverse curriculum
    increment_frequency = 100
    sample_window_width = 25
    increment = 25
    use_demo_prob = 0.5
    # params for adaptive curriculum
    mixing = ["reverse"]
    mixing_ratio = [1]
    ratio_step = [0]
    curriculum_episodes = 20
    improve_threshold = 0.1
    curriculum_length = 50
    history_length = 10


env = make(
    "SawyerPickPlace",
    ignore_done=True,
    use_camera_obs=False,
    reward_shaping=True,
    single_object_mode=1,
    demo_config=demo_config,
)

for _ in range(20):
    env.reset()
    env.render()
    time.sleep(1)
    for i in range(100000):
        obs, reward, done, _ = env.step(np.zeros(env.dof))
        print(reward)
        env.render()
