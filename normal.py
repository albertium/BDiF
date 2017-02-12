"""
Non sorting version using dict as hash table
"""

from mpi4py import MPI
import numpy as np
import json
import datetime
import os
import sys
from scipy.stats import chi2


def log(vb, code, msg):
    """
    :param vb: (verbosity) 0 - Basic, 1 - Routine, 2 - Debug
    :param code: (msg category) 0 - Info, 1 - Warning, 2 - Error
    :param msg:
    :return: None
    """

    if vb <= verbose or code == 2:
        codex = {0: "INFO", 1: "WARNING", 2: "ERROR"}
        msg = "[{}][{}] ({}): {}\n"\
            .format(rank, codex[code], datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), msg)
        flog.Write_shared(bytearray(msg.encode()))


def remove_file(loc):
    try:
        os.remove(loc)
    except:
        pass


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

    # start MPI and get basic info
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    num_nodes = comm.Get_size()

    ####################################################################################################################
    # read parameters
    try:
        with open("config", "r") as f:
            params = json.load(f)

        chunk_size = params["chunk_size"]
        chunk_buf = params["chunk_buf"]
        verbose = params["verbose"]
        input_file = params["input_file"]
        normal_file = params["normal_file"]
        log_freq = params["log_freq"]
        enable_timing = params["enable_timing"]

        is_param_loaded = True

    except:
        is_param_loaded = False

    ####################################################################################################################
    # reset log, signal and noise file
    if rank == 0:
        remove_file("log_norm")
        remove_file(normal_file)

    # sychronize threads
    flag = comm.bcast(1, root=0)

    flog = MPI.File.Open(comm, "log_norm", MPI.MODE_WRONLY | MPI.MODE_CREATE)
    if is_param_loaded:
        log(0, 0, "Parameters loaded")
    else:
        log(0, 2, "Failed to load parameters")
        sys.exit(1)

    ####################################################################################################################
    # read file in chunks
    fh = MPI.File.Open(comm, input_file, MPI.MODE_RDONLY)

    file_size = fh.Get_size()
    block_size = int(file_size / num_nodes)
    block_offset = rank * block_size
    num_chunks = int(np.ceil(block_size / chunk_size))

    log(0, 0, "Block size is {} bytes, divided into {} chunks".format(block_size, num_chunks))

    ####################################################################################################################
    # main loop
    track_time = [datetime.timedelta(0)] * 4  # Read time, filter time, write time, misc time
    tic()  # start timing

    x_n = 0
    x_stats = np.zeros(4)  # correspond to mean, mean squared, mean cubic, and mean quad

    for iteration in range(num_chunks):
        # read in a chunk of data
        if block_size % chunk_size > 0 and iteration == num_chunks - 1:
            buf = bytearray(block_size % chunk_size)
        else:
            buf = bytearray(chunk_size)

        offset = block_offset + iteration * chunk_size
        log(1, 0, "Reading chunk {} from {:,} to {:,}".format(iteration, offset, offset + chunk_size))

        # ---- checkpoint ---- #
        track_time[3] += toc()

        handle = fh.Read_at(offset, buf)
        if handle:
            log(1, 2, "Failed to read data: {}".format(handle))

        # ---- checkpoint ---- #
        track_time[0] += toc()  # read time

        # discard everything before the first "\n"; if last chunk, discard the last record
        price = np.array([float(p)
                          for row in buf[:chunk_size].decode().split("\n")[1:-1]
                          for x, p, y in [row.split(",")]
                          if 0 < float(p) < 50000])

        # update statistics
        tmp_n = x_n + price.size
        x_stats = x_n/(tmp_n) * x_stats + price.size/(tmp_n) * np.array(list(map(lambda x: np.mean(price**x), range(1, 5))))
        x_n = tmp_n

        # ---- checkpoint ---- #
        track_time[1] += toc()  # filter time


    all_stats = comm.gather((x_n, x_stats), root=0)
    if rank == 0:
        f_n = np.sum([n[0] for n in all_stats])
        f_stats = np.zeros(4)
        for stat in all_stats:
            f_stats += stat[0]/f_n * stat[1]

        sq = np.sqrt(f_stats[1] - np.power(f_stats[0], 2))
        skew = (f_stats[2] - 3*f_stats[0]*f_stats[1] + 2*np.power(f_stats[0], 3)) / np.power(sq, 3)
        kurt = (f_stats[3] - 4*f_stats[0]*f_stats[2] + 6*f_stats[0]*f_stats[0]*f_stats[1] - 3*np.power(f_stats[0], 4)) \
               / np.power(sq, 4)
        JB = (skew*skew + 0.25*np.power(kurt-3, 2))

        # to avoid overflow
        if f_n > 10000 and JB > 1:
            JB = 10000
            msg = "(this number is truncated to avoid overflow)"
        else:
            JB = f_n / 6 * JB
            msg = ""

        p_value = 1 - chi2.cdf(JB, 2)

        # write result to file
        with open(normal_file, "w") as f:
            f.write("Skewness:{:.5f}\nKurtosis:{:.5f}\nJB Statistic:{:.3f}{}\np-value:{:.6f}\n"
                    .format(skew, kurt, JB, msg, p_value))
            f.write("Normal:{}".format(bool(p_value > 0.05)))

    # close files
    flog.Close()
    fh.Close()



