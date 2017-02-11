from datetime import datetime
import numpy as np

with open("data/data-big.txt", "r") as f:
    raw = f.read()

rows = raw.split("\n")
parsed = [(datetime.strptime(x[0], "%Y%m%d:%H:%M:%S.%f"), float(x[1]), int(x[2])) for row in rows if row for x in [row.split(",")]]
print(parsed[:10])

parsed_sorted = sorted(parsed, key=lambda x: x[0])
print(parsed_sorted[:10])

# duplicates
noise = []
signal = []
for i in range(len(parsed_sorted)-1):
    if parsed_sorted[i] == parsed_sorted[i+1]:
        noise.append(i)
    elif parsed_sorted[i][1] <= 0 or parsed_sorted[i][1] > 10000:
        noise.append(i)
    elif parsed_sorted[i][2] <= 0:
        noise.append(i)
    else:
        signal.append(i)

print("noise: " + str(len(noise)))
print("signal: " + str(len(signal)))

tmp = [parsed_sorted[ix] for ix in noise]
noise_series = [",".join([datetime.strftime(x[0], "%Y%m%d:%H:%M:%S.%f"), "{:.2f}".format(x[1]), str(x[2])]) for x in tmp]

print(noise_series[-2:])

with open("output/noise_benchmark.txt", "w") as f:
    f.write("\n".join(noise_series) + "\n")

# with open("output/noise.txt", "r") as f:
#     raw = f.read()
#
# to_compare = raw.split("\n")
# print(len(to_compare))
# wrong = [row for row in to_compare if row not in noise_series]
# print(len(wrong))