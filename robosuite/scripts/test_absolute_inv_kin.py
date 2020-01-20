import os
from robosuite import make
import numpy as np
from robosuite.utils.ffmpeg_gif import save_gif

def compute_inv_kin(pos, rotation):
    from robosuite.controllers.sawyer_ik_controller import SawyerIKController
    import robosuite
    bullet_data_path = os.path.join(robosuite.models.assets_root, "bullet_data")
    ik_controller = SawyerIKController(bullet_data_path, lambda : [0, -1.18, 0.00, 2.18, 0.00, 0.57, 3.3161])

    # convert from target pose in base frame to target pose in bullet world frame
    world_targets = ik_controller.bullet_base_pose_to_world_pose((pos, rotation))
    rest_poses = [0, -1.18, 0.00, 2.18, 0.00, 0.57, 3.3161]

    for bullet_i in range(100):
        arm_joint_pos = ik_controller.inverse_kinematics(
            world_targets[0], world_targets[1], rest_poses=rest_poses
        )
        ik_controller.sync_ik_robot(arm_joint_pos, sync_last=True)

    return arm_joint_pos


def render_traj_from_states():
    frames = []

    env_name = 'SawyerPickPlaceBread'

    env = make(
        env_name,
        has_renderer=False,
        ignore_done=True,
        use_camera_obs=True,
        use_object_obs=True,
        camera_height=64,
        camera_width=64,
    )

    eef_pose = np.array([
        [5.57672847e-01, -3.75483733e-02,  9.82644815e-01, -9.99996896e-01, -2.21727214e-03,  1.13446425e-03, -7.32291109e-05],
        [5.61621856e-01,  2.88248548e-02,  1.05441987e+00, -9.99980409e-01,  3.48495165e-03, -5.12371593e-03, -8.85872802e-04]
    ])

    for i in range(eef_pose.shape[0]):

        eef_pos = eef_pose[i, :3]
        eef_quat = eef_pose[i, 3:]
        joint_pos = compute_inv_kin(eef_pos, eef_quat)

        sim_state = env.sim.get_state()
        qpos = sim_state.qpos

        for i, j in enumerate(env._ref_joint_pos_indexes):
            qpos[j] = joint_pos[i]
        # for i, j in enumerate(env._ref_gripper_joint_pos_indexes):
        #     qpos[j] = gripper_pos[i]

        sim_state.qpos[:] = qpos

        env.sim.set_state(sim_state)
        env.sim.forward()

        obs = env._get_observation()
        frame = obs["image"][::-1]
        frames.append(frame)
        print('commanded endeff pos', eef_pos)
        print('achieved endeff pos', obs['eef_pos'])
        print('delta comm-achieved endeff pos', eef_pos - obs['eef_pos'])

        if len(frames) == 10:
            break

    save_gif("test_inv_kin.gif", frames, fps=15)


if __name__ == '__main__':
    render_traj_from_states()

