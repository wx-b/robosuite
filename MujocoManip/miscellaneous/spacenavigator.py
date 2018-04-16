# Mac driver for SpaceNav controller
from __future__ import print_function

import hid
import time
import threading
import numpy as np

from collections import namedtuple

import MujocoManip.miscellaneous.utils as U

AxisSpec = namedtuple('AxisSpec', ['channel', 'byte1', 'byte2', 'scale'])

SpNavSpec = {
    "x": AxisSpec(channel=1, byte1=1, byte2=2, scale=1),
    "y": AxisSpec(channel=1, byte1=3, byte2=4, scale=-1),
    "z": AxisSpec(channel=1, byte1=5, byte2=6, scale=-1),
    "roll": AxisSpec(channel=1, byte1=7, byte2=8, scale=-1),
    "pitch": AxisSpec(channel=1, byte1=9, byte2=10, scale=-1),
    "yaw": AxisSpec(channel=1, byte1=11, byte2=12, scale=1),
}

# convert two 8 bit bytes to a signed 16 bit integer
def to_int16(y1, y2):
    x = (y1) | (y2<<8)
    if x>=32768:
        x = -(65536-x)
    return x

def scale_to_control(x, axis_scale=350.):
    x = x / axis_scale
    x = min(max(x, -1.0), 1.0)
    return x

def convert(b1, b2):
    return scale_to_control(to_int16(b1, b2))


class SpaceNavigator(object):

    def __init__(self,
                 vendor_id=9583,
                 product_id=50735,):

        print("Opening SpaceNavigator device")
        print(hid.enumerate())
        self.device = hid.device()
        self.device.open(vendor_id, product_id) # SpaceNavigator

        print("Manufacturer: %s" % self.device.get_manufacturer_string())
        print("Product: %s" % self.device.get_product_string())

        self.double_click_and_hold = False
        self.single_click_and_hold = False

        # launch daemon thread to listen to SpaceNav
        self.thread = threading.Thread(target=self.run)
        self.thread.daemon = True
        self.thread.start()

        self._control = [0, 0, 0, 0, 0, 0]

        self.rotation = np.array([[-1., 0., 0.], [0., 1., 0.], [0., 0., -1.]])

    def get_controller_state(self):
        """
        Returns the current state of the 3d mouse, a dictionary of pos, orn, and grasp.
        """
        dpos = self.control[:3] * 0.005
        roll, pitch, yaw = self.control[3:] * 0.005
        self.grasp = self.control_gripper

        # convert RPY to an absolute orientation
        drot1 = U.rotation_matrix(angle=-pitch, direction=[1., 0., 0.], point=None)[:3, :3]
        drot2 = U.rotation_matrix(angle=roll, direction=[0., 1., 0.], point=None)[:3, :3]
        drot3 = U.rotation_matrix(angle=yaw, direction=[0., 0., 1.], point=None)[:3, :3]
        self.rotation = self.rotation.dot(drot1.dot(drot2.dot(drot3)))

        return dict(dpos=dpos, rotation=self.rotation, 
                    grasp=self.grasp)

    def run(self):

        t_last_click = -1
        t_last_release = -1

        while True:
            d = self.device.read(13)
            if d is not None:
                # print('read: "{}"'.format(d))

                if d[0] == 1:
                    self.y = convert(d[1], d[2])
                    self.x = convert(d[3], d[4])
                    self.z = convert(d[5], d[6]) * -1.0

                    self.roll = convert(d[7], d[8])
                    self.pitch = convert(d[9], d[10])
                    self.yaw = convert(d[11], d[12])

                    self._control = [self.x, self.y, self.z,
                                     self.roll, self.pitch, self.yaw]

                elif d[0] == 3:

                    # press left button
                    if d[1] == 1:

                        t_click = time.time()
                        elapsed_time = t_click - t_last_click
                        t_last_click = t_click
                        self.single_click_and_hold = True
                        if elapsed_time < 0.3:
                            self.double_click_and_hold = True

                    # release left button
                    if d[1] == 0:
                        self.single_click_and_hold = False
                        self.double_click_and_hold = False

                    # save right button for future purpose
                    if d[1] == 2: pass

    @property
    def control(self):
        """
        Returns 6-DoF control
        """
        return np.array(self._control)

    @property
    def control_gripper(self):
        if self.double_click_and_hold:
            return -1.0
        elif self.single_click_and_hold:
            return 1.0
        else:
            return 0

if __name__ == '__main__':

    spacenav = SpaceNavigator()
    for i in range(100):
        print(spacenav.control, spacenav.control_gripper())
        time.sleep(0.02)