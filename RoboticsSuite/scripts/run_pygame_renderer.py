"""pygame rendering demo.

This script provides an example of using the pygame library for rendering
camera observations as an alternative to the default mujoco_py renderer.

"""

import sys
import pygame
import numpy as np

import RoboticsSuite


if __name__ == "__main__":

    width = 512
    height = 384
    screen = pygame.display.set_mode((width, height))

    env = RoboticsSuite.make(
        "BaxterLift",
        has_renderer=False,
        ignore_done=True,
        camera_height=height,
        camera_width=width,
        show_gripper_visualization=True,
        use_camera_obs=True,
        use_object_obs=False,
        use_eef_ctrl=False,
    )

    for i in range(10000):

        # issue random actions
        action = 0.5 * np.random.randn(env.dof)
        obs, reward, done, info = env.step(action)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()

        # read camera observation
        im = np.flip(obs["image"].transpose((1, 0, 2)), 1)
        pygame.pixelcopy.array_to_surface(screen, im)
        pygame.display.update()

        if i % 100 == 0:
            print("step #{}".format(i))

        if done:
            break
