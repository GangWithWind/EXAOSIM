import ctypes
import numpy as np

ll = ctypes.cdll.LoadLibrary

lib = ll("./lib/libtestcu.so")


ai = np.array([1, 2, 3], dtype=np.int32)
al = np.array([1, 2, 3], dtype=np.int64)
af = np.array([1, 2, 3], dtype=np.float32)

lib.array_trans_i(ai.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)), 3)
print(ai)

lib.array_trans_l(al.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)), 3)
print(al)

lib.array_trans_f(af.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), 3)
print(af)
# print(lib.foo(10))

# print(lib.set_global(20))
# print(lib.get_global())
# print(lib.set_global(30))
# print(lib.get_global())