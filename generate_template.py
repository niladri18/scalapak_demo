import os
import sys

if __name__=="__main__":


    M = 1024
    N = 1024
    K = 1024
    Mb = 64
    Nb = 64

    nodes = 8
    ntask_per_node = 4
    nprow = 8
    npcol = 4

    nprocs = nodes*ntask_per_node
    ncores = int(128/ntask_per_node)

    if nprow*npcol != nprocs:
        raise ValueError("Processors and grids dont match")

    l = [0]*11

    l[0] = "#!/bin/bash"
    l[1] = "#SBATCH -C cpu"
    l[2] = "#SBATCH --qos=debug"
    l[3] = "#SBATCH --time=00:10:00"
    l[4] = f"#SBATCH --nodes={nodes}"
    l[5] = f"#SBATCH --ntasks-per-node={ntask_per_node}"
    l[6] = f"#SBATCH --output=job_{nodes}x{ntask_per_node}_{M}_{N}_{K}_{Mb}_{Nb}_{nprow}_{npcol}.out"
    l[7] = ""
    l[8] = ""
    l[9] = ""

    l[10] = f"srun -n {nprocs} -c {ncores} --cpu-bind=cores ./scaling {M} {N} {K} {Mb} {Nb} {nprow} {npcol}"

    fname = "job_template.sh"
    f = open(fname,"w")
    for i in range(11):
        print(l[i])
        print(l[i], file = f)

    f.close()



