
<!-- GETTING STARTED -->
## Getting Started

This work explores matrix multiplication in a distributed architecture.


### Prerequisites


Check out [this Youtube link](https://www.youtube.com/watch?v=Jgvoks1RWB0) to install MPI and scalapack.


* Compile:

  ```sh
  mpicxx test.cpp -L/usr/lib -lscalapack -lblas -llapack -lgfortran -o matmul
  ```

* Run:

  ```sh
  mpirun -n 8 ./mat_vec <M> <N> <K> <Mb> <Nb> <Npx> <Npy>
  ```

   In the processor grid:

  `Npx`: number of processors along x axis
  `Npy`: number of processors along y axis

### Description of files 

(relevant files only)

`scaling.cpp` matrix multiplication $`C = AB`$ using scalapack

`dft_app.cpp` matrix multiplication $`C = A^{T}B`$ using  scalapack

`custom.cpp` matrix multiplication $`C = AB`$  (SUMMA) using MPI without scalapack and cblacs

`mat_vec.cpp` matrix vector multiplication $`C = AX`$ using elemental cyclic distribution 

### Literature


[PARALLEL MATRIX MULTIPLICATION: A SYSTEMATIC JOURNEY](https://www.cs.utexas.edu/users/flame/pubs/2D3DFinal.pdf)
