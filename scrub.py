from mpi4py import MPI
import numpy as np
import json
import datetime
import os, subprocess


def log(vb, code, msg):
    '''
    :param vb: (verbosity) 0 - Basic, 1 - Routine, 2 - Debug
    :param code: (msg category) 0 - Info, 1 - Warning, 2 - Error
    :param msg:
    :return: None
    '''
    if vb <= verbose or code == 2:
        codex = {0: "INFO", 1: "WARNING", 2: "ERROR"}
        msg = "[" + str(rank) + "][" + codex[code] + "]" + " (" + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") \
              + "): " + msg + "\n"
        flog.Write_shared(bytearray(msg.encode()))

def count_line(loc):
    '''
    :param loc: location of target file
    :return: number of lines in target file
    '''
    cmd = "wc -l " + loc
    process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
    output, err = process.communicate()

    if output:
        return int(output.decode().split()[0])
    else:
        return -1

if __name__ == "__main__":

    # start MPI and get basic info
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    num_nodes = comm.Get_size()

    # reset log file
    if rank == 0:
        try:
            os.remove("log")
        except:
            pass

    flog = MPI.File.Open(comm, "log", MPI.MODE_WRONLY | MPI.MODE_CREATE)

    ####################################################################################################################
    # read parameters
    try:
        with open("config", "r") as f:
            params = json.load(f)

        chunk_size = params["chunk_size"]
        chunk_buf = params["chunk_buf"]
        overlap_size = params["overlap_size"]
        verbose = params["verbose"]

        log(0, 0, "Loaded in parameters")

    except:
        log(0, 2, "Fail to load in parameters")

    ####################################################################################################################
    # read file in chunks
    fh = MPI.File.Open(comm, "data/data-small.txt", MPI.MODE_RDONLY)
    fsig = MPI.File.Open(comm, "output/signal.txt", MPI.MODE_WRONLY | MPI.MODE_CREATE)
    fnoi = MPI.File.Open(comm, "output/noise.txt", MPI.MODE_WRONLY | MPI.MODE_CREATE)

    file_size = fh.Get_size()
    block_size = int(file_size / num_nodes)
    block_offset = rank * block_size

    log(1, 0, "Block size is " + str(block_size) + " bytes")

    ####################################################################################################################
    num_chunks = int(block_size // chunk_size)
    overlap = []  # store overlap parsed records
    overlap_raw = []  # store overlap raw records

    log(1, 0, "Read data in " + str(num_chunks) + " chunks")
    for iter in range(num_chunks):
        # read in a chunk of data
        buf = bytearray(chunk_size + chunk_buf)
        offset = block_offset + iter * chunk_size
        log(1, 0, "Reading chunk " + str(iter) + " from " + str(offset) + " to " + str(offset + chunk_size + chunk_buf))

        handle = fh.Read_at(offset, buf)
        if handle:
            log(1, 2, "Failed to read data: " + handle)

        # discard everything before the first "\n"
        rows = buf[:chunk_size].decode().split("\n")[1:]

        # get remaining part from the remainder
        extra = buf[chunk_size:].decode().split("\n")[0]
        # if last byte is "\n", take one more record from the remainder
        if buf[chunk_size - 1] == 10:
            rows.append(extra)
        else:
            # append the remaining part of last row from the remainder
            rows[-1] += extra

        parsed = overlap + [(x[0], float(x[1]), int(x[2])) for row in rows for x in [row.split(",")]]
        rows = overlap_raw + rows
        # sort by time
        sorted_ix = np.argsort([x[0] for x in parsed])

        log(2, 0, "Row size: " + str(len(rows)) + "/" + str(len(overlap_raw)) + ". Parse size: " + str(len(parsed)) +
            "/" + str(len(overlap)))

        ################################################################################################################

        signal = []
        noise = []
        for i, j in zip(sorted_ix[:-overlap_size], sorted_ix[1:-(overlap_size-1)]):
            # filter by duplicates, price anomaly and negative volume
            if parsed[i] == parsed[j] or parsed[i][1] <= 0 or parsed[i][1] > 50000 or parsed[i][2] <= 0:
                noise.append(i)
            else:
                signal.append(i)

        print(len(parsed), len(rows))
        overlap = [parsed[ix] for ix in sorted_ix[-overlap_size:]]
        overlap_raw = [rows[ix] for ix in sorted_ix[-overlap_size:]]

        # output result
        fsig.Write_shared(bytearray("\n".join([rows[ix] for ix in signal]).encode()))
        fnoi.Write_shared(bytearray("\n".join([rows[ix] for ix in noise]).encode()))

    # insanity check
    ori_size = count_line("data/data-small.txt")
    sig_size = count_line("output/signal.txt")
    noi_size = count_line("output/noise.txt")
    log(1, 0, "Noise: " + str(noi_size) + ". Difference: " + str(ori_size - sig_size - noi_size))

    # close files
    flog.Close()
    fh.Close()
    fsig.Close()
    fnoi.Close()


