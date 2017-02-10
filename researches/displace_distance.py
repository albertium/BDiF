import numpy as np

with open("../data-small.txt", "r") as f:
    raw = f.read()

rows = raw.split("\n")[:-1]
parsed = [x for row in rows for x, y, z in [row.split(",")]]
displace = np.argsort(parsed) - range(len(parsed))
print(np.max(displace))
print(np.sum(displace < 100)/len(parsed))