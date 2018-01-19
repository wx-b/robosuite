from dm_control.mujoco import Physics as dm_Physics
import ctypes

c_double_p = ctypes.POINTER(ctypes.c_double)
NULL = c_double_p()


class Physics(dm_Physics):
    pass
    # TODO: make it work so we have jacobians
    # def get_body_jacp(self, name):
    #     return
    #     body_id = self.named.model.id(name)
    #     jacp = np.zeros(3 * self.model.nv)
    #     mj_jacBody(self._model.ptr, self.ptr, jacp.ctypes.data_as(c_double_p), NULL, body_id)
    #     return jacp

    # def get_body_jacr(self, name):
    #     pass
    #     # body_id = self.model.named.body_id(name)
    #     # cdef double * jacr_view = &jacr[0]
    #     # mj_jacBody(self._model.ptr, self.ptr, NULL, jacr_view, body_id)
    #     # return jacr