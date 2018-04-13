from mujoco_py import MjViewer
from mujoco_py.generated import const
import glfw

class MujocoPyRenderer():
    def __init__(self, sim):
        """
        sim should be MjSim
        """
        self.viewer = MjViewer(sim)

    def set_camera(self, camera_id):
        """
        Set the camera view to the specified camera ID.
        """
        self.viewer.cam.fixedcamid = camera_id
        self.viewer.cam.type = const.CAMERA_FIXED

    def render(self, *args, **kwargs):
        # safe for multiple calls
        self.viewer.render()

    def close(self):
        """
        Destroys the open window and renders (pun intended) the viewer useless.
        """
        glfw.destroy_window(self.viewer.window)
        self.viewer = None


