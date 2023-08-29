
<!-- GETTING STARTED -->
## Getting Started

This is an example of how you may give instructions on setting up your project locally.
To get a local copy up and running follow these simple example steps.

### Prerequisites

This is an example of how to list things you need to use the software and how to install them.
* npm
  ```sh
  mpicxx test.cpp -L/usr/lib -lscalapack -lblas -llapack -lgfortran -o matmul
  ```

### Description of files

`scaling.cpp` matrix multiplication `$C = AB$` using scalapack

`dft_app.cpp` matrix multiplication $`C = A^{T}B`$ using  scalapack
`custom.cpp` matrix multiplication `$C = AB$`  (SUMMA) using MPI without scalapack and cblacs

