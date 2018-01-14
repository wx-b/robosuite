from mujoco_py import MjSim, MjViewer
class MujocoPyRenderer():
    def __init__(self, sim):
        """
        sim should be MjSim
        """
        self.viewer = MjViewer(sim)

    def render(self, *args, **kwargs):
        # safe for multiple calls
        self.viewer.render()