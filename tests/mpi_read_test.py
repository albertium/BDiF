from mpi4py import MPI

comm = MPI.COMM_WORLD
mode = MPI.MODE_RDONLY
fh = MPI.File.Open(comm, "../config", mode)

# set read size parameter
file_size = fh.Get_size()
num_process = comm.Get_size()
rank = comm.Get_rank()
chunk_size = int(file_size / num_process)
start = rank * chunk_size + 1

# read all
buf = bytearray(50)
fh.Read_at(0, buf)

raw = buf.decode()
print(buf)

buf = bytearray(50)
fh.Read_at(50, buf)
print(buf)
# print("Process", rank, "finish reading. Total size is", len(raw), "bytes.")