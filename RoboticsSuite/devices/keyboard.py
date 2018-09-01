"""
Driver class for Keyboard controller.

Must run `pip install pynput` to use this device.
"""

import os
import numpy as np
from pynput.keyboard import Key, Listener, Controller
from RoboticsSuite.devices import Device
from RoboticsSuite.utils.transform_utils import rotation_matrix

class Keyboard(Device):
    """A minimalistic driver class for a Keyboard."""

    def __init__(self):
        """
        Initialize a Keyboard device.
        """
        if os.name != 'nt' and os.geteuid() != 0:
            exit("You need to have root privileges to use a keyboard device.\nPlease try again, this time using 'sudo'. Exiting.")

        self._display_controls()
        self._reset_internal_state()

        self._reset_state = 0
        self._enabled = False
        self._pos_step = 0.05

        # make a thread to listen to keyboard and register our callback functions
        self.listener = Listener(on_press=self.on_press, on_release=self.on_release)

        # start listening
        self.listener.start()

    def _display_controls(self):
        """
        Method to pretty print controls.
        """
        def print_command(char, info):
            char += " " * (10 - len(char))
            print("{}\t{}".format(char, info))
        print("")
        print_command("key", "command")
        print_command("e", "enable/disable control")
        print_command("q", "reset the simulation")
        print_command("spacebar", "open/close the gripper")
        print_command("w-a-s-d", "move arm in a horizontal plane")
        print_command("r-f", "move arm vertically")
        print_command("z-x", "rotate arm along x-axis")
        print_command("t-g", "rotate arm along y-axis")
        print_command("c-v", "rotate arm along z-axis")

    def _reset_internal_state(self):
        """
        Resets internal state of controller, except for the reset signal.
        """
        self.rotation = np.array([[-1., 0., 0.], [0., 1., 0.], [0., 0., -1.]])
        self.pos = np.zeros(3) # (x, y, z)
        self.last_pos = np.zeros(3)
        self.grasp = False

    def start_control(self):
        """
        Method that should be called externally before controller can 
        start receiving commands. 
        """
        self._reset_internal_state()
        self._reset_state = 0
        self._enabled = True

    def get_controller_state(self):
        """Returns the current state of the keyboard, a dictionary of pos, orn, grasp, and reset."""
        dpos = self.pos - self.last_pos
        self.last_pos = np.array(self.pos)
        return dict(dpos=dpos, rotation=self.rotation, grasp=int(self.grasp), reset=self._reset_state)

    def on_press(self, key):
        """
        Key handler for key presses.
        """

        # print('{} pressed'.format(key))

        # note that some keys don't have the "char" attribute
        # this might lead to an AttributeError
        try:
            # controls for moving position
            if key.char == "w":
                self.pos[0] -= self._pos_step # dec x
            elif key.char == "s":
                self.pos[0] += self._pos_step # inc x
            elif key.char == "a":
                self.pos[1] -= self._pos_step # dec y
            elif key.char == "d":
                self.pos[1] += self._pos_step # inc y
            elif key.char == "f":
                self.pos[2] -= self._pos_step # dec z
            elif key.char == "r":
                self.pos[2] += self._pos_step # inc z

            # controls for moving orientation
            elif key.char == "z":
                drot = rotation_matrix(angle=0.1, direction=[1., 0., 0.], point=None)[:3, :3]
                self.rotation = self.rotation.dot(drot) # rotates x 
            elif key.char == "x":
                drot = rotation_matrix(angle=-0.1, direction=[1., 0., 0.], point=None)[:3, :3]
                self.rotation = self.rotation.dot(drot) # rotates x
            elif key.char == "t":
                drot = rotation_matrix(angle=0.1, direction=[0., 1., 0.], point=None)[:3, :3]
                self.rotation = self.rotation.dot(drot) # rotates y
            elif key.char == "g":
                drot = rotation_matrix(angle=-0.1, direction=[0., 1., 0.], point=None)[:3, :3]
                self.rotation = self.rotation.dot(drot) # rotates y
            elif key.char == "c":
                drot = rotation_matrix(angle=0.1, direction=[0., 0., 1.], point=None)[:3, :3]
                self.rotation = self.rotation.dot(drot) # rotates z
            elif key.char == "v":
                drot = rotation_matrix(angle=-0.1, direction=[0., 0., 1.], point=None)[:3, :3]
                self.rotation = self.rotation.dot(drot) # rotates z

        except AttributeError as e:
            pass

    def on_release(self, key):
        """
        Key handler for key releases.
        """

        # print('{0} release'.format(key))

        try:
            # controls for grasping
            if key == Key.space:
                self.grasp = not self.grasp # toggle gripper 

            # user-commanded reset
            elif key.char == "q":
                self._reset_state = 1
                self._enabled = False
                self._reset_internal_state()


        except AttributeError as e:
            pass


if __name__ == "__main__":
    pass
