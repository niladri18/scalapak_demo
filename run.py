import subprocess
import time


n = 5
output = [0]*n
Mb = 32
Nb = 32
print(f"Blocking: {Mb}x{Nb}")

for i in range(1,n+1):
    t1 = time.time()
    output[i-1] = subprocess.run(["mpirun", "-n", str(2**(i+1)), "./scaling", "1024", "1024","1024",str(Mb),str(Nb)],\
                        stdout = subprocess.PIPE, universal_newlines = True).stdout
    t2 = time.time()
    print(f"Nproc: {2**(i+1)} Finished in {(t2-t1)*1000} millisec")
    #print(2**(i+1))

