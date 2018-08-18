"""Demo of learning curriculum utilities.

Several prior works have demonstrated the effectiveness of altering the
start state distribution of training episodes for learning RL policies.
We provide a generic utility for setting learning curriculums. These
curricula can be either constructed from demonstrations of the task,
or with a fixed/adaptive strategy.

Related work:

[1] Reinforcement and Imitation Learning for Diverse Visuomotor Skills
Yuke Zhu, Ziyu Wang, Josh Merel, Andrei Rusu, Tom Erez, Serkan Cabi,Saran Tunyasuvunakool,
János Kramár, Raia Hadsell, Nando de Freitas, Nicolas Heess
RSS 2018

[2] Backplay: "Man muss immer umkehren"
Cinjon Resnick, Roberta Raileanu, Sanyam Kapoor, Alex Peysakhovich, Kyunghyun Cho, Joan Bruna
arXiv:1807.06919

[3] DeepMimic: Example-Guided Deep Reinforcement Learning of Physics-Based Character Skills
Xue Bin Peng, Pieter Abbeel, Sergey Levine, Michiel van de Panne
Transactions on Graphics 2018

[4] Approximately optimal approximate reinforcement learning
Sham Kakade and John Langford
ICML 2002
"""

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
