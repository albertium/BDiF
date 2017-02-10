import json
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
fh = MPI.File.Open(comm, "config", MPI.MODE_RDONLY)
file_size = fh.Get_size()
buf = bytearray(file_size)
store = buf.decode()
fh.Iread_at(0, buf)
# fh.Close()
# params = json.loads(buf.decode())

# with open("config", "r") as f:
#     params = json.load(f)

print("Process", rank, "finish", "Answer: ", "no")