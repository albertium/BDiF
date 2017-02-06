##############################################
#       Basic utility functions
##############################################

def read_small():
    with open("./data/data-small.txt", "r") as f:
        for line in f:
            line = line.strip().split(",")
            yield(line)