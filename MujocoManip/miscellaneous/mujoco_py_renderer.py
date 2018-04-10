from mujoco_py import MjViewer
import glfw

class MujocoPyRenderer():
    def __init__(self, sim):
        """
        sim should be MjSim
        """
        self.viewer = MjViewer(sim)

    def render(self, *args, **kwargs):
        # safe for multiple calls
        self.viewer.render()

    def close(self):
        """
        Destroys the open window and renders (pun intended) the viewer useless.
        """
        glfw.destroy_window(self.viewer.window)
        self.viewer = None


