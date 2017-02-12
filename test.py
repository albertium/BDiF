
from mpi4py import MPI
from time import sleep
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

data = np.ones(10) * rank
data = comm.gather(data, root=0)

if rank == 0:
    print(data)



