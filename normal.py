"""
Non sorting version using dict as hash table
"""

from mpi4py import MPI
import numpy as np
import json
import datetime
import os
import sys
import psutil
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

    p_id = psutil.Process(os.getpid())

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
    track_time = np.array([datetime.timedelta(0)] * 4)  # Read time, parse time, cal time, misc time per "log_freq" iterations
    track_time_final = np.array([datetime.timedelta(0)] * 4)  # total track time
    track_mem = np.array([0, np.inf, 0])  # for max, min and mean memory usage
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

        # ---- checkpoint ---- #
        track_time[1] += toc()  # parse time

        # update statistics
        tmp_n = x_n + price.size
        x_stats = x_n/tmp_n*x_stats + price.size/tmp_n*np.array(list(map(lambda x: np.mean(price**x), range(1, 5))))
        x_n = tmp_n

        # ---- checkpoint ---- #
        track_time[2] += toc()  # calculation time

        # record memory usage
        mem = p_id.memory_info().rss
        if mem > track_mem[0]:
            track_mem[0] = mem  # for max memory
        if mem < track_mem[1]:
            track_mem[1] = mem  # for min memory
        track_mem[2] += mem  # for mean

        if (iteration+1) % log_freq == 0:
            # log processing time
            track_time_final += track_time
            track_time /= log_freq
            log(0, 0, "Chunk {} to {} processing time: read={:.3f}s parse={:.3f}s calculation={:.3f}s misc={:.3f}s"
                .format(iteration-log_freq+1, iteration, *[x.total_seconds() for x in track_time]))

            # log processing memory usage
            track_mem[2] /= log_freq
            track_mem /= 2 ** 20
            log(0, 0, "Chunk {} to {} memory usage: max={:.1f} MB min={:.1f} MB mean={:.1f} MB"
                .format(iteration-log_freq+1, iteration, *track_mem))
            track_mem = np.array([0, np.inf, 0])  # reset for next "log_freq" iterations

    # collect statistic from all the threads
    all_stats = comm.gather((x_n, x_stats), root=0)
    if rank == 0:
        # aggregate results
        f_n = np.sum([n[0] for n in all_stats])
        f_stats = np.zeros(4)
        for stat in all_stats:
            f_stats += stat[0]/f_n * stat[1]

        # calculate statistics - standard deviation, skewness, kurtosis
        sq = np.sqrt(f_stats[1] - np.power(f_stats[0], 2))
        skew = (f_stats[2] - 3*f_stats[0]*f_stats[1] + 2*np.power(f_stats[0], 3)) / np.power(sq, 3)
        kurt = (f_stats[3] - 4*f_stats[0]*f_stats[2] + 6*f_stats[0]*f_stats[0]*f_stats[1] - 3*np.power(f_stats[0], 4)) \
               / np.power(sq, 4)
        JB = f_n / 6 * (skew*skew + 0.25*np.power(kurt-3, 2))
        p_value = 1 - chi2.cdf(JB, 2)

        # write result to file
        with open(normal_file, "w") as f:
            f.write("Skewness:{:.5f}\nKurtosis:{:.5f}\nJB Statistic:{:.3f}\np-value:{:.6f}\n"
                    .format(skew, kurt, JB, p_value))
            f.write("Normal:{}".format(bool(p_value > 0.05)))

    # close files
    flog.Close()
    fh.Close()



