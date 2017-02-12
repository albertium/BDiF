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
import subprocess


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


def count_line(loc):
    """
    :param loc: location of target file
    :return: number of lines in target file
    """

    cmd = "wc -l " + loc
    process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
    output, err = process.communicate()

    if output:
        return int(output.decode().split()[0])
    else:
        return -1


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
        overlap_size = params["overlap_size"]
        verbose = params["verbose"]
        input_file = params["input_file"]
        signal_file = params["signal_file"]
        noise_file = params["noise_file"]
        log_freq = params["log_freq"]
        enable_timing = params["enable_timing"]

        is_param_loaded = True

    except:
        is_param_loaded = False

    ####################################################################################################################
    # reset log, signal and noise file
    if rank == 0:
        remove_file("log")
        remove_file("output/signal.txt")
        remove_file("output/noise.txt")

    # sychronize threads
    flag = comm.bcast(1, root=0)

    flog = MPI.File.Open(comm, "log", MPI.MODE_WRONLY | MPI.MODE_CREATE)
    if is_param_loaded:
        log(0, 0, "Parameters loaded")
    else:
        log(0, 2, "Failed to load parameters")
        sys.exit(1)

    fsig = MPI.File.Open(comm, signal_file, MPI.MODE_WRONLY | MPI.MODE_CREATE)
    fnoi = MPI.File.Open(comm, noise_file, MPI.MODE_WRONLY | MPI.MODE_CREATE)

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
    tot_sig = 0  # total number of signals
    tot_noi = 0  # total number of noise
    track_counts = [0, 0, 0]  # check progress
    track_time = [datetime.timedelta(0)] * 4  # Read time, filter time, write time, misc time
    track_mem = np.array([0, np.inf, 0])  # for max, min and mean memory usage
    tic()  # start timing

    for iteration in range(num_chunks):
        # read in a chunk of data
        if block_size % chunk_size > 0 and iteration == num_chunks - 1:
            buf = bytearray(block_size % chunk_size)
        else:
            buf = bytearray(chunk_size + chunk_buf)

        offset = block_offset + iteration * chunk_size
        log(1, 0, "Reading chunk {} from {:,} to {:,}".format(iteration, offset, offset + chunk_size + chunk_buf))

        # ---- checkpoint ---- #
        track_time[3] += toc()
        handle = fh.Read_at(offset, buf)
        if handle:
            log(1, 2, "Failed to read data: {}".format(handle))

        # ---- checkpoint ---- #
        track_time[0] += toc()  # read time

        # discard everything before the first "\n"; if last chunk, discard the last record
        if iteration == num_chunks - 1:
            rows = buf[:chunk_size].decode().split("\n")[1:-1]
        else:
            rows = buf[:chunk_size].decode().split("\n")[1:]

            # get remaining part from the remainder if not the last chunk
            extra = buf[chunk_size:].decode().split("\n")[0]
            # if last byte is "\n", take one more record from the remainder
            if buf[chunk_size - 1] == 10:
                rows[-1] = extra
            else:
                # otherwise, append the remaining part of last row from the remainder
                rows[-1] += extra

        # ---- checkpoint ---- #
        track_time[3] += toc()

        signal = []
        noise = []
        hash_table = {}
        for row in rows:
            # if duplicate, remove
            if row in hash_table:
                noise.append(row)
                continue
            hash_table[row] = True

            t, p, v = row.split(",")
            if float(p) <= 0 or float(p) > 10000 or int(v) <= 0:
                noise.append(row)
            else:
                signal.append(row)

        # ---- checkpoint ---- #
        track_time[1] += toc()  # filter time

        # output result
        msg = "\n".join(signal) + "\n"
        fsig.Write_shared(bytearray(msg.encode()))
        msg = "\n".join(noise) + "\n"
        fnoi.Write_shared(bytearray(msg.encode()))

        # ---- checkpoint ---- #
        track_time[2] += toc()  # write time

        # logging
        track_counts[0] += len(rows)
        track_counts[1] += len(signal)
        track_counts[2] += len(noise)

        tot_sig += len(signal)
        tot_noi += len(noise)

        # ---- checkpoint ---- #
        track_time[3] += toc()

        # record memory usage
        mem = p_id.memory_info().rss
        if mem > track_mem[0]:
            track_mem[0] = mem  # for max memory
        if mem < track_mem[1]:
            track_mem[1] = mem  # for min memory
        track_mem[2] += mem  # for mean

        if (iteration + 1) % log_freq == 0:
            log(0, 0, "Processed chunk {}. Cumulatively {:,} records, {:,} signal and {:,} noise"
                .format(iteration, *track_counts))
            track_counts = [0, 0, 0]

            # log processing memory usage
            track_mem[2] /= log_freq
            track_mem /= 2 ** 20
            log(0, 0, "Chunk {} to {} memory usage: max={:.1f} MB min={:.1f} MB mean={:.1f} MB"
                .format(iteration - log_freq + 1, iteration, *track_mem))
            track_mem = np.array([0, np.inf, 0])  # reset for next "log_freq" iterations


    # insanity check
    log(0, 0, "Read:{:.2f}s Filter:{:.2f}s Write:{:.2f}s Misc:{:.2f}s"
        .format(*[x.total_seconds() for x in track_time]))

    ori_size = count_line(input_file)
    log(0, 0, "Noise:{:,} Signal:{:,}".format(tot_noi, tot_sig, ori_size - tot_sig - tot_noi))

    comm.gather(1, root=0)
    if rank == 0:
        sig_size = count_line(signal_file)
        noi_size = count_line(noise_file)
        log(0, 0, "Total noise:{:,} Total signal:{:,} Difference:{:,}"
            .format(noi_size, sig_size, ori_size - sig_size - noi_size))

    # close files
    flog.Close()
    fh.Close()
    fsig.Close()
    fnoi.Close()


