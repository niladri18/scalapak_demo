#include <iostream>
#include <cstdlib>
#include <cmath>
#include <mpi.h>
#include <iomanip>
#include <string>
#include <fstream>
#include <sstream>
#include <mutex>
#include <vector>
#include <limits.h>
#include <cassert>
#include <chrono>
#include <fstream>
#include <memory>

extern "C" {
  int dgemm_(char *, char *, int *, int *, int *, double *, double *, int *,
              double *, int *, double *, double *, int *);
}

void readMatrixFromFile2(const std::string& filename, double*& matrix, int* numRows, int* numCols) {
    std::ifstream file(filename);
    if (file.is_open()) {
        //file >> *numRows >> *numCols;
	int ndim = (*numRows) * (*numCols);

        //matrix = new double [ndim];
        for (int i = 0; i < ndim; i++) {
                file >> matrix[i];
		//std::cout<<matrix[i]<<std::endl;
        }

        file.close();
    } else {
        std::cout << "Unable to open file: " << filename << std::endl;
    }
}


void readMatrixFromFile(const std::string& filename, double*& matrix, int* numRows, int* numCols) {
    std::ifstream file(filename);
    if (file.is_open()) {
        //file >> *numRows >> *numCols;
	int ndim = (*numRows) * (*numCols);

        //matrix = new double [ndim];
        for (int i = 0; i < ndim; i++) {
                file >> matrix[i];
		std::cout<<matrix[i]<<"  ";
        }
	std::cout<<std::endl;

        file.close();
    } else {
        std::cout << "Unable to open file: " << filename << std::endl;
    }
}

void resetMatrix(double* matrix, int numRows, int numCols) {
    for (int i = 0; i < numRows*numCols; i++) {
            matrix[i] = 0;
    }
}

void initMatrix(double*& matrix, int numRows, int numCols){
  //matrix = new double [numRows*numCols];
  for (int i = 0; i < numRows*numCols; i++){
      //matrix[i] = (double) (rand() / ((double) (INT_MAX * 1.0)));
      matrix[i] = (i)%numCols + 1;
  }
}

void printMatrix(double* matrix, int numRows, int numCols) {
    for (int i = 0; i < numRows; i++) {
        for (int j = 0; j < numCols; j++) {
            std::cout << matrix[i*numCols + j] << " ";
        }
        std::cout << std::endl;
    }
}

void writeMatrix(std::string fname, double* matrix, int numRows, int numCols) {
    std::ofstream outputFile(fname);
    for (int i = 0; i < numRows; i++) {
        for (int j = 0; j < numCols; j++) {
            outputFile << matrix[i*numCols + j] << " ";
        }
        outputFile << std::endl;
    }
    outputFile.close();
}

template<typename T>
class Matrix{
    public:
    int row;
    int col;
    T *matrix = nullptr;

    Matrix(int row, int col): row(row),col(col){
        matrix = new T[row*col];
        for (int i = 0; i< row*col; i++){
            matrix[i] = 0;
        }
    }
    ~Matrix(){
        delete matrix;
    }
    void initMatrix(){
        for (int i = 0; i< row*col; i++){
            matrix[i] = 0;
        }
    }

        void print(){
        for (int i = 0; i< row; i++){
            for (int j = 0; j < col; j++){
                std::cout<<matrix[i*col+j]<<" ";
            }
            std::cout<<"\n";
        }
    }
    void read(std::string &fname){
        std::ifstream input(fname);
        T d;
        int i = 0;
        while(input>>d){
            std::cout<<d<<" ";
            matrix[i] = d;
            i+= 1;
        }
        std::cout<<"\n";
    }
    void random_init(){
        for (int i = 0; i < row*col; i++){
            //matrix[i] = (double) (rand() / ((double) (INT_MAX * 1.0)));
            matrix[i] = (i)%col + 1;
        }
        

    }


};

int main(int argc, char* argv[]) {
  // Initialize mpi
  MPI_Init(&argc, &argv);
  int M = atoi(argv[1]);
  int N = atoi(argv[2]);
  int K = atoi(argv[3]);
  int Mb = atoi(argv[4]);
  int Nb = atoi(argv[5]);
  int nprow = atoi(argv[6]);
  int npcol = atoi(argv[7]);

  int p_id = 7;
  // Get the number of processors and their ids
  /* Begin Cblas context */
  int myrank, nprocs;
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
  //Cblacs_pinfo(&myrank, &nprocs);
  //blacs_pinfo_(&myrank, &nprocs);
  //printf("Process id %d of %d\n", myrank, nprocs);


  // cast num of processors into a rectangular grid (nprow x npcol)
  /* We assume that we have nprocs processes and place them in a (nprow , npcol) grid */
  //int nprow = static_cast<int>(std::sqrt(nprocs));
  //int npcol = nprocs / nprow;
  int context;
  int i0 = 0;
  int i1 = 0;

  //Cblacs_get(0, 0, &context);
  //Cblacs_gridinit(&context, "Row-major", nprow, npcol);

  // create 2d grid communicators
  MPI_Comm world_comm;
  int ierr;
  int dims[2];
  dims[0] = nprow;
  dims[1] = npcol;
  int wrap_around[2];
  wrap_around[0] = 0;
  wrap_around[1] = 0;
  ierr = MPI_Cart_create( MPI_COMM_WORLD, 2, dims, wrap_around, 0, &world_comm);
  if(ierr != 0) printf("ERROR[%d] creating CART\n",ierr);
  //printf("Proc row col: %d %d \n",nprow, npcol);

  /* Print grid pattern */
  int myrow, mycol;
  int coord[2];
  MPI_Cart_coords(world_comm, myrank, 2, coord);
  bool mpiroot = (myrank == 0);
  myrow = coord[0];
  mycol = coord[1];
  //printf("Proc row col: %d %d \n",myrow, mycol);

  int my_cart_rank;
  MPI_Cart_rank(world_comm, coord, &my_cart_rank);

  /* create row-wise communicators */
  MPI_Comm row_comm;
  int color_a = my_cart_rank/nprow;
  //Determine color based on row
  MPI_Comm_split(world_comm, color_a, my_cart_rank, &row_comm);
  int row_rank, row_size;
  MPI_Comm_rank(row_comm, &row_rank);
  MPI_Comm_size(row_comm, &row_size);
  //printf("CART RANK/WORLD RANK: %d/%d \t ROW_COMM RANK/SIZE: %d/%d\n",
	//my_cart_rank, myrank, row_rank, row_size);

  /* create col-wise communicators */
  MPI_Comm col_comm;
  int color_b = my_cart_rank/npcol;
  //Determine color based on col
  MPI_Comm_split(world_comm, color_b, my_cart_rank, &col_comm);
  int col_rank, col_size;
  MPI_Comm_rank(col_comm, &col_rank);
  MPI_Comm_size(col_comm, &col_size);
  //printf("CART RANK/WORLD RANK: %d/%d \t COL_COMM RANK/SIZE: %d/%d\n",
	//my_cart_rank, myrank, col_rank, col_size);


  
  /* Read the matrices */
  double *A_glob = new double [M*K];
  double *B_glob = new double [K*N];
  double *C_glob = new double [M*N];
  //initMatrix(C_glob, M, N);
  if (mpiroot) {
    initMatrix(A_glob, M, K);
    printMatrix(A_glob,M,K);
    //printMatrix(A_glob, M, K);
    initMatrix(B_glob, K, N);
    printMatrix(B_glob,K,N);
    std::cout<<"====================="<<std::endl;
    //printf("A(%d,%d); blocking: %d,%d\n",M,K, Mb, Nb);
  }
  //printMatrix(A_glob,M,K);
  MPI_Bcast(&M, 1, MPI_INT, 0, world_comm);
  MPI_Bcast(&K, 1, MPI_INT, 0, world_comm);
  MPI_Bcast(&N, 1, MPI_INT, 0, world_comm);
  resetMatrix(C_glob, M, N);
  MPI_Bcast(A_glob, M*K, MPI_DOUBLE, 0, world_comm);
  MPI_Bcast(B_glob, K*N, MPI_DOUBLE, 0, world_comm);
  MPI_Bcast(C_glob, M*N, MPI_DOUBLE, 0, world_comm);


  MPI_Barrier(world_comm);
  double *worka = new double [M*1];
  double *workb = new double [1*N];
  double *workc = new double [M*N];

  resetMatrix(worka, M, 1);
  resetMatrix(workb, 1, N);
  resetMatrix(workc, M, N);

  double alpha = 1, beta = 1;

  int common = 1, one = 1, zero = 0;
  char transA = 'N', transB = 'N';


    // cast each row of A along nprows:
  int myM = M/nprow; // size of the row in each proc
  double *myworka = new double [myM*1];
  resetMatrix(myworka, myM,1);

  int myN = N/npcol; // size of the row in each proc
  double *myworkb = new double [1*myN];
  resetMatrix(myworkb, 1,myN);

  double *myworkc = new double [myM*myN];
  resetMatrix(myworkc, myM,myN);

  for(int k=0; k<K; k++){
        for(int i=0; i<M;i++){
            worka[i] = A_glob[i*K + k  ];
        }
    
        // distribute local worka along rows

        MPI_Scatter(worka, myM, MPI_DOUBLE, myworka, myM, MPI_DOUBLE,0,row_comm);

    // cast B along npcols:

        for(int i=0; i<N;i++){
            workb[i] = B_glob[i + k*N ];
        }
    
        // distribute local worka along rows

    
        MPI_Scatter(workb, myN, MPI_DOUBLE, myworkb, myN, MPI_DOUBLE,0,col_comm);

        /*
        if (mpiroot){
            printf("-----\n");
            printMatrix(myworka,myM,1);
            printf("-----\n");
            printMatrix(myworkb,1,myN);
        }
        */

        //dgemm_(&transA, &transB , &myM, &myN, &common, &alpha, myworka, &myM, myworkb, &common, &beta, myworkc, &myM);

        dgemm_(&transB, &transA , &myN, &myM, &common, &alpha, myworkb, &myN, myworka, &common, &beta, myworkc, &myN);
  }



  /*
  for(int i = 0; i < nprocs; ++i) {
  MPI_Barrier(MPI_COMM_WORLD);
  if (myrank == i){
  printf("C in (%d,%d)\n", myrow, mycol);
  printMatrix(myworkc, myM, myN);
  fflush(stdout);
  }
  }
  */

  MPI_Barrier(MPI_COMM_WORLD);
  // collect 
  for (int i = 0; i < myM; i++){
      int ii = myrow*myM + i;
      for(int  j = 0; j<myN; j++){
          int jj = mycol*myN + j;
          C_glob[ii*N + jj] = myworkc[i*myN + j];
          //printf("(%d,%d) ",ii,jj);
          //std::cout<<myworkc[i*myN + j]<<" ";
      }

  }

  MPI_Allreduce(MPI_IN_PLACE, C_glob, M*N, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  if (mpiroot){
    printMatrix(C_glob, M,N);
  }

  delete[] A_glob;
  delete[] B_glob;
  delete[] C_glob;

  MPI_Comm_free(&world_comm);
  MPI_Finalize();

  return 0;
}

