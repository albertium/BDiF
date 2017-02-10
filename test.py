import subprocess

process = subprocess.Popen("wc -l data/signal.txt".split(), stdout=subprocess.PIPE)
output, err = process.communicate()
print(output)
print(err)
if output:
    print("no")