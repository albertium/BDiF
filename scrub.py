from mpi4py import MPI
import numpy as np
import json
import datetime
import os
import sys
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
    overlap = []  # store overlap parsed records
    overlap_raw = []  # store overlap raw records
    tot_sig = 0  # total number of signals
    tot_noi = 0  # total number of noise
    track_counts = [0, 0, 0]  # check progress
    track_time = [datetime.timedelta(0)] * 6  # Read time, parse time, sort time, filter time, write time, misc time
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
        track_time[5] += toc()
        handle = fh.Read_at(offset, buf)
        if handle:
            log(1, 2, "Failed to read data: {}".format(handle))

        # ---- checkpoint ---- #
        track_time[0] += toc()

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
        track_time[5] += toc()

        try:
            parsed = overlap + [(x[0], float(x[1]), int(x[2])) for row in rows for x in [row.split(",")]]
        except:
            log(1, 2, "Parsing failed at chunk {}".format(iteration))
            with open("output/dump.txt", "w") as f:
                f.write("\n".join(rows))
            break

        rows = overlap_raw + rows

        # ---- checkpoint ---- #
        track_time[1] += toc()  # parse time

        # sort by time
        sorted_ix = np.argsort([x[0] for x in parsed])

        # ---- checkpoint ---- #
        track_time[2] += toc()  # sort time

        log(2, 0, "Row size: " + str(len(rows)) + "/" + str(len(overlap_raw)) + ". Parse size: " + str(len(parsed)) +
            "/" + str(len(overlap)))

        ################################################################################################################

        signal = []
        noise = []

        # if not the last chunk, leave overlap for next chunk; otherwise, process the whole chunk
        if iteration == num_chunks - 1:
            end_ix = len(parsed)
        else:
            end_ix = len(parsed) - overlap_size

        # ---- checkpoint ---- #
        track_time[5] += toc()

        for i, j in zip(sorted_ix[:end_ix], sorted_ix[1:(end_ix + 1)]):
            # filter by duplicates, price anomaly and negative volume
            if parsed[i] == parsed[j] or parsed[i][1] <= 0 or parsed[i][1] > 50000 or parsed[i][2] <= 0:
                noise.append(i)
            else:
                signal.append(i)

        # ---- checkpoint ---- #
        track_time[3] += toc()  # filter time

        # leave overlap if not the last chunk
        if iteration < num_chunks - 1:
            overlap = [parsed[ix] for ix in sorted_ix[-overlap_size:]]
            overlap_raw = [rows[ix] for ix in sorted_ix[-overlap_size:]]

        # ---- checkpoint ---- #
        track_time[5] += toc()

        # output result
        msg = "\n".join([rows[ix] for ix in signal]) + "\n"
        fsig.Write_shared(bytearray(msg.encode()))
        msg = "\n".join([rows[ix] for ix in noise]) + "\n"
        fnoi.Write_shared(bytearray(msg.encode()))

        # ---- checkpoint ---- #
        track_time[4] += toc()  # write time

        # logging
        track_counts[0] += end_ix
        track_counts[1] += len(signal)
        track_counts[2] += len(noise)

        tot_sig += len(signal)
        tot_noi += len(noise)

        # ---- checkpoint ---- #
        track_time[5] += toc()

        if (iteration + 1) % log_freq == 0:
            log(0, 0, "Processed chunk {}. Cumulatively {:,} records, {:,} signal and {:,} noise"
                .format(iteration, *track_counts))
            track_counts = [0, 0, 0]


    # insanity check
    log(0, 0, "Read:{:.2f}s Parse:{:.2f}s Sort:{:.2f}s Filter:{:.2f}s Write:{:.2f}s Misc:{:.2f}s"
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


