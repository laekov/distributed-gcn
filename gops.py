import os
from ctypes import cdll, POINTER, c_float, c_ulong, c_void_p, c_int


ops_lib = cdll.LoadLibrary(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'ops.so'))

init = ops_lib.init

ops_lib.allred.argtypes = [c_void_p, c_void_p, c_ulong]
allreduce = ops_lib.allred

ops_lib.rank.restype = c_int
get_rank = ops_lib.rank

ops_lib.size.restype = c_int
get_size = ops_lib.size

ops_lib.ring_pass.argtypes = [c_void_p, c_void_p, c_ulong, c_int]
ring_pass = ops_lib.ring_pass

wait = ops_lib.wait
finalize = ops_lib.finalize
