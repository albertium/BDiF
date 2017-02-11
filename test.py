
# from mpi4py import MPI
# from time import sleep
# import os
# import sys
#
# comm = MPI.COMM_WORLD
# rank = comm.Get_rank()
#
# fr = MPI.File.Open(comm, "data/data-big.txt", MPI.MODE_RDONLY)
# fw = MPI.File.Open(comm, "write_test.txt", MPI.MODE_WRONLY | MPI.MODE_CREATE)
#
# print(fr.Get_size())
#
# buf = bytearray(fr.Get_size())
# request = fr.Iread_at(0, buf)
# request.wait()
#
# rows = [row for row in buf[:1000000].decode().split("\n")]
# for i in range(10):
#     msg = "\n".join(rows[-10:]) + "\n"
#     fw.Write_shared(bytearray(msg.encode()))
#     sleep(0.2)
#
#
#
# fr.Close()
# fw.Close()

import datetime
import time

def tic():
    global watch
    watch = datetime.datetime.now()

def toc():
    global watch
    curr = datetime.datetime.now()
    duration = curr - watch
    watch = curr
    return duration

if __name__ == "__main__":
    tic()
    a = [datetime.timedelta(0)]
    time.sleep(0.5)
    a[0] += toc()
    time.sleep(0.7)
    print(toc())
    time.sleep(0.1)
    print(a[0].total_seconds())