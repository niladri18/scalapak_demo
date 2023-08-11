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

extern "C" {
  void blacs_get_(int*, int*, int*);
  void blacs_pinfo_(int*, int*);
  void blacs_gridinit_(int*, char*, int*, int*);
  void blacs_gridinfo_(int*, int*, int*, int*, int*);
  void Cblacs_pinfo(int* myrank, int* nprocs);
  void Cblacs_get(int context, int request, int* value);
  void Cblacs_gridinit(int* context, const char* order, int nprow, int npcol);
  void Cblacs_gridinfo(int context, int* nprow, int* npcol, int* myrow, int* mycol);
  void Cblacs_gridexit(int context);
  void Cblacs_barrier(int context, char* scope);
  void Cblacs_pcoord(int context, int pnum,int* prow, int* pcol);
  int numroc_(int *n, int *nb, int *iproc, int *srcproc, int *nprocs);
  int indxg2p_(int const& glob, int const& nb, int const& iproc, int const& isproc, int const& nprocs);
  int indxl2g_(int const& loc, int const& nb, int const& iproc, int const& isproc, int const& nprocs);



  void descinit_(int* desc, int* m, int* n, int* mb, int* nb, int* irsrc, int* icsrc, int* ictxt, int* lld, int* info);
  void pdgemm_(const char* transa, const char* transb, int* m, int* n, int* k, double* alpha, double* a, int* ia, int* ja, int* desca, double* b, int* ib, int* jb, int* descb, double* beta, double* c, int* ic, int* jc, int* descc);
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
      matrix[i] = (double) (rand() / ((double) (INT_MAX * 1.0)));
      //matrix[i] = i/numCols + 1;
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
int main(int argc, char* argv[]) {
    /* Computes A^T B*/

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
  Cblacs_pinfo(&myrank, &nprocs);
  //blacs_pinfo_(&myrank, &nprocs);
  //printf("Process id %d of %d\n", myrank, nprocs);

  bool mpiroot = (myrank == 0);

  // cast num of processors into a rectangular grid (nprow x npcol)
  /* We assume that we have nprocs processes and place them in a (nprow , npcol) grid */
  //int nprow = static_cast<int>(std::sqrt(nprocs));
  //int npcol = nprocs / nprow;
  int context;
  int i0 = 0;
  int i1 = 0;

  Cblacs_get(0, 0, &context);
  Cblacs_gridinit(&context, "Row-major", nprow, npcol);

  //printf("Proc row col: %d %d \n",nprow, npcol);

  /* Print grid pattern */
  int myrow, mycol;
  //Cblacs_pcoord(context, myrank, &myrow, &mycol);
  Cblacs_gridinfo(context, &nprow, &npcol, &myrow, &mycol);

  /* Read the matrices */


  double *A_glob = nullptr;
  double *B_glob = nullptr;
  double *C_glob = nullptr;
  
  A_glob = new double [K*M];
  B_glob = new double [K*N];
  C_glob = new double [M*N];
  //initMatrix(C_glob, M, N);
  if (mpiroot) {
    //std::string Afilename = "example.dat";
    std::string Afilename = "A3.TXT";
    //readMatrixFromFile2(Afilename, A_glob, &M, &K);
    initMatrix(A_glob, K, M);
    //printMatrix(A_glob,M,K);
    //printMatrix(A_glob, M, K);
    std::string Bfilename = "B3.TXT";
    //std::string Bfilename = "new2.dat";
    //readMatrixFromFile2(Bfilename, B_glob, &K, &N);
    initMatrix(B_glob, K, N);
    printf("A(%d,%d) ; B(%d,%d); blocking: %d,%d\n",M,K,N,K, Mb, Nb);
    //printMatrix(B_glob,K,N);
    //printf("A(%d,%d); blocking: %d,%d\n",M,K, Mb, Nb);
  }
  if (mpiroot){
    //printf("C \n");
    //printMatrix(C_glob, M, N);
    writeMatrix("A.TXT", A_glob, K, M);
    writeMatrix("B.TXT", B_glob, K, N);
  }
  //printMatrix(A_glob,M,K);
  MPI_Bcast(&M, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&K, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
  //MPI_Bcast(&A_glob[0], M*K, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(A_glob, K*M, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  //MPI_Bcast(&B_glob[0], K*N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(B_glob, K*N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  //printf("A(%d,%d) ; B(%d,%d) blocking [%d,%d] in PID %d\n",M,K,K,N, Mb, Nb, myrank);
  //MPI_Barrier(MPI_Comm communicator);
  //printMatrix(C_glob,M,N);
  MPI_Barrier(MPI_COMM_WORLD);


  // divide the matrix blocks among the processors
  int izero_a, izero_b,izero_c, iZERO;
  izero_a = 0;
  izero_b = 0;
  izero_c = 0;
  iZERO = 0;
  int zero, one;
  zero = 0;
  one = 1;

  //how big is "my" chunk of matrix A, B, C
  int row_a = numroc_(&K, &Mb, &myrow, &zero, &nprow);
  int col_a = numroc_(&M, &Nb, &mycol, &zero, &npcol);

  // B is distributed along columns
  int row_b = numroc_(&K, &Mb, &myrow, &zero, &nprow);
  int col_b = numroc_(&N, &Nb, &mycol, &zero, &npcol);

  int row_c = numroc_(&M, &Mb, &myrow, &zero, &nprow);
  int col_c = numroc_(&N, &Nb, &mycol, &zero, &npcol);

  //printf("B-dim in Pid %d: grid[%d,%d] : (%d,%d) \n",myrank, myrow, mycol, row_b,col_b);
  //printf("C-dim in Pid %d: grid[%d,%d] : (%d,%d) \n",myrank, myrow, mycol, row_c,col_c);




  /* Initialize local arrays */

  double* localA = new double [row_a*col_a];

  double* localB = new double [row_b*col_b];

  double* localC = new double [row_c*col_c];

  // convert local index to global index in block-cyclic distribution
  int ii, jj;

  for(int my_j = 0; my_j < col_a; my_j++){
      	jj = (((my_j/Nb) * npcol) + mycol)*Mb + my_j%Mb;
 	for(int my_i = 0; my_i < row_a; my_i++){
      		ii = (((my_i/Mb) * nprow) + myrow)*Mb + my_i%Mb;
		localA[my_j*row_a + my_i] = A_glob[ii*M + jj];
		//localA[my_i + my_j*row_a] = A_glob[ii*K + jj];
      }
  }




  //printf("A-dim in Pid %d: grid[%d,%d] : (%d,%d) \n",myrank, myrow, mycol, row_a,col_a);
  /*
  printf("local A in Pid: %d\n", myrank);
  for(int i = 0; i < nprocs; ++i) {
	  MPI_Barrier(MPI_COMM_WORLD);
  if (myrank == i){
	    printf("A in (%d,%d)\n", myrow, mycol);
  	    printMatrix(localA, row_a, col_a);
  	    fflush(stdout);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    }

    */

for(int my_j = 0; my_j < col_b; my_j++){
    jj = (((my_j/Nb) * npcol) + mycol)*Mb + my_j%Mb; 
 	for(int my_i = 0; my_i < row_b; my_i++){
      		ii = (((my_i/Mb) * nprow) + myrow)*Mb + my_i%Mb; 
		localB[my_j*row_b + my_i] = B_glob[ii*N + jj]; 
      }
  }



  /*
  for(int i = 0; i < nprocs; ++i) {
  MPI_Barrier(MPI_COMM_WORLD);
  if (myrank == i){
  printf("B in (%d,%d)\n", myrow, mycol);
  printMatrix(localB, row_b, col_b);
  fflush(stdout);
  }
  MPI_Barrier(MPI_COMM_WORLD);
  }
  */


  



  /* Prepare array descriptors for ScaLAPACK */
  int desc_a[9], desc_b[9], desc_c[9];
  int info;
  descinit_( desc_a,  &K, &M, &Mb, &Nb, &zero, &zero, &context, &row_a, &info);
  descinit_( desc_b,  &K, &N, &Mb, &Nb, &zero, &zero, &context, &row_b, &info);
  descinit_( desc_c,  &M, &N, &Mb, &Nb, &zero, &zero, &context, &row_c, &info);


  double alpha = 1.0;
  double beta = 0.0;


  //for(int i = 0; i < nprocs; ++i) {
  //MPI_Barrier(MPI_COMM_WORLD);
  //if (myrank == i){
  //printf("C in (%d,%d)\n", myrow, mycol);
  //printMatrix(localC, row_c, col_c);
  //fflush(stdout);
  //}
  //}
  //pdgemm_("N", "N", &row_a, &col_c, &col_a, &alpha, localA,  &one, &one, desc_a, localB, &one, &one, desc_b, &beta, localC, &one, &one, desc_c);
  int n_cold_run = 2;
  int n_hot_run = 8;

  std::vector<double> time_list;
  


  for (int i = 0; i < n_cold_run; i++){
  	pdgemm_("T", "N", &M, &N, &K, &alpha, &localA[0],  &one, &one, desc_a, &localB[0], &one, &one, desc_b, &beta, &localC[0], &one, &one, desc_c);

  }

  for (int i = 0; i < n_hot_run; i++ ){
  	std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();

  	pdgemm_("T", "N", &M, &N, &K, &alpha, &localA[0],  &one, &one, desc_a, &localB[0], &one, &one, desc_b, &beta, &localC[0], &one, &one, desc_c);
  	std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
  	std::chrono::steady_clock::duration timeTaken = end - start;
	time_list.push_back(std::chrono::duration_cast<std::chrono::milliseconds>(timeTaken).count());
  }

  MPI_Barrier(MPI_COMM_WORLD);
  double tmax = 0;
  double tmean = 0;
  double tmin = 1000000; 

  for ( auto v : time_list){
	  //std::cout<<v<<std::endl;
	  tmean += v;
	  if(v > tmax){
		  tmax = v;
	  }
	  if (v < tmin){
		  tmin = v;
	  }
  }
  tmean /= n_hot_run;
  if(mpiroot){
  	printf("%d\t%f\t%f\t%f\t%d\t%d\n",nprocs, tmean, tmax, tmin, nprow, npcol);
  }


  //for(int i = 0; i < nprocs; ++i) {
  //MPI_Barrier(MPI_COMM_WORLD);
  //if (myrank == i){
  //printf("C in (%d,%d)\n", myrow, mycol);
  //printMatrix(localC, row_c, col_c);
  //fflush(stdout);
  //}
  //}

  resetMatrix(C_glob, M, N);
  //if (myrank == 0){
  //printf("C in (%d,%d)\n", myrow, mycol);
  //printMatrix(localC, row_c, col_c);
  //fflush(stdout);
  //}

  // Collect local C:
  
    for(int my_j = 0; my_j < col_c; my_j++){
        jj = (((my_j/Nb) * npcol) + mycol)*Mb + my_j%Mb; 
 	    for(int my_i = 0; my_i < row_c; my_i++){
      		ii = (((my_i/Mb) * nprow) + myrow)*Mb + my_i%Mb; 
            C_glob[ii*N + jj] = localC[my_j*row_c + my_i];
        }
    }
  

  //printf("C matrix assigned\n");

  int MPI_Barrier(MPI_Comm communicator);
  MPI_Allreduce(MPI_IN_PLACE, C_glob, M*N, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  int MPI_Barrier(MPI_Comm communicator);

  if (mpiroot){
    //printf("C \n");
    //printMatrix(C_glob, M, N);
    writeMatrix("ans.out", C_glob, M, N);
  }
  //if (mpiroot){
  //std::cout << "Time taken: " << std::chrono::duration_cast<std::chrono::milliseconds>(timeTaken).count()
  //            << " milliseconds" << std::endl;
  //}
  delete[] localA;
  delete[] localB;
  delete[] localC;

  Cblacs_gridexit(context);
  MPI_Finalize();

  return 0;
}

