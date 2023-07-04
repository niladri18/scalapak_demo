#include <iostream>
#include <cstdlib>
#include <cmath>
#include <mpi.h>
#include <iomanip>
#include <string>
#include <fstream>
#include <sstream>
#include <mutex>
//#include <scalapack.h>


extern "C" {
  void Cblacs_pinfo(int* myrank, int* nprocs);
  void Cblacs_get(int context, int request, int* value);
  void Cblacs_gridinit(int* context, const char* order, int nprow, int npcol);
  void Cblacs_gridinfo(int context, int* nprow, int* npcol, int* myrow, int* mycol);
  void Cblacs_gridexit(int context);
  void Cblacs_barrier(int context, char* scope);
  void Cblacs_pcoord(int context, int pnum,int* prow, int* pcol);
  int numroc_(int *n, int *nb, int *iproc, int *srcproc, int *nprocs);

  void descinit_(int* desc, int* m, int* n, int* mb, int* nb, int* irsrc, int* icsrc, int* ictxt, int* lld, int* info);
  void pdgemm_(const char* transa, const char* transb, int* m, int* n, int* k, double* alpha, double* a, int* ia, int* ja, int* desca, double* b, int* ib, int* jb, int* descb, double* beta, double* c, int* ic, int* jc, int* descc);
}

void readMatrixFromFile(const std::string& filename, double**& matrix, int* numRows, int* numCols) {
    std::ifstream file(filename);
    if (file.is_open()) {
        file >> *numRows >> *numCols;

        matrix = new double*[ *numRows];
        for (int i = 0; i < *numRows; i++) {
            matrix[i] = new double[*numCols];
            for (int j = 0; j < *numCols; j++) {
                file >> matrix[i][j];
            }
        }

        file.close();
    } else {
        std::cout << "Unable to open file: " << filename << std::endl;
    }
}


void readMatrixFromFile2(const std::string& filename, double**& matrix, int* numRows, int* numCols) {
    std::ifstream file(filename);
    if (file.is_open()) {
        //file >> *numRows >> *numCols;

        //matrix = new double*[ *numRows];
        for (int i = 0; i < *numRows; i++) {
            //matrix[i] = new double[*numCols];
            for (int j = 0; j < *numCols; j++) {
                file >> matrix[i][j];
            }
        }

        file.close();
    } else {
        std::cout << "Unable to open file: " << filename << std::endl;
    }
}

void resetMatrix(double** matrix, int numRows, int numCols) {
    for (int i = 0; i < numRows; i++) {
        for (int j = 0; j < numCols; j++) {
            matrix[i][j] = 0;
        }
    }
}

void initMatrix(double** matrix, int numRows, int numCols){
  matrix = new double* [numRows];
  for (int i = 0; i < numRows; i++){
      matrix[i] = new double [numCols];
  }

}

void printMatrix(double** matrix, int numRows, int numCols) {
    for (int i = 0; i < numRows; i++) {
        for (int j = 0; j < numCols; j++) {
            std::cout << matrix[i][j] << " ";
        }
        std::cout << std::endl;
    }
}

int main(int argc, char** argv) {
  // Initialize mpi
  MPI_Init(&argc, &argv);

  // Get the number of processors and their ids
  /* Begin Cblas context */
  int myrank, nprocs;
  Cblacs_pinfo(&myrank, &nprocs);
  //printf("Process id %d of %d\n", myrank, nprocs);

  bool mpiroot = (myrank == 0);

  // cast num of processors into a rectangular grid (nprow x npcol)
  /* We assume that we have nprocs processes and place them in a (nprow , npcol) grid */
  int nprow = static_cast<int>(std::sqrt(nprocs));
  int npcol = nprocs / nprow;
  int context;
  int zero = 0;
  Cblacs_get(0, 0, &context);
  //Cblacs_get(&zero, &zero, &context);
  Cblacs_gridinit(&context, "Row-major", nprow, npcol);
  printf("Proc row col: %d %d \n",nprow, npcol);

  /* Print grid pattern */
  int myrow, mycol;
  //Cblacs_pcoord(context, myrank, &myrow, &mycol);
  Cblacs_gridinfo(context, &nprow, &npcol, &myrow, &mycol);

  /* Read the matrices */


  double **A_glob = nullptr;
  double **B_glob = nullptr;
  double **C_glob = nullptr;
  int M = 0, N=0, K=0, Mb = 2, Nb = 2 ;
  initMatrix(A_glob, M, K);
  initMatrix(B_glob, K, N);
  if (mpiroot) {
    std::string Afilename = "example.dat";
    readMatrixFromFile(Afilename, A_glob, &M, &K);
    //printMatrix(A_glob, M, K);
    std::string Bfilename = "example.dat";
    readMatrixFromFile(Bfilename, B_glob, &K, &N);
    printf("A(%d,%d) ; B(%d,%d); blocking: %d,%d\n",M,K,K,N, Mb, Nb);
    //printf("A(%d,%d); blocking: %d,%d\n",M,K, Mb, Nb);
  }
  MPI_Bcast(&M, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&K, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
  printf("A(%d,%d) ; B(%d,%d) blocking [%d,%d] in PID %d\n",M,K,K,N, Mb, Nb, myrank);


  // divide the matrix blocks among the processors
  int izero_a, izero_b,izero_c, iZERO;
  izero_a = 0;
  izero_b = 0;
  izero_c = 0;
  iZERO = 0;
  //how big is "my" chunk of matrix A, B, C
  int row_a = numroc_(&M, &Mb, &myrow, &izero_a, &nprow);
  int col_a = numroc_(&K, &Nb, &mycol, &izero_a, &npcol);

  int row_b = numroc_(&K, &Mb, &myrow, &izero_b, &nprow);
  int col_b = numroc_(&N, &Nb, &mycol, &izero_b, &npcol);

  int row_c = numroc_(&M, &Mb, &myrow, &izero_c, &nprow);
  int col_c = numroc_(&N, &Nb, &mycol, &izero_c, &npcol);

  printf("A-dim in Pid %d: grid[%d,%d] : (%d,%d) \n",myrank, myrow, mycol, row_a,col_a);
  printf("B-dim in Pid %d: grid[%d,%d] : (%d,%d) \n",myrank, myrow, mycol, row_b,col_b);
  printf("C-dim in Pid %d: grid[%d,%d] : (%d,%d) \n",myrank, myrow, mycol, row_c,col_c);




  /* Initialize local arrays */

  double** localA = new double* [row_a];
  for (int i = 0; i < row_a; i++){
      localA[i] = new double [col_a];
  }

  double** localB = new double* [row_b];
  for (int i = 0; i < row_b; i++){
      localB[i] = new double [col_b];
  }

  double** localC = new double* [row_c];
  for (int i = 0; i < row_c; i++){
      localC[i] = new double [col_c];
  }

  // convert local index to global index in block-cyclic distribution
  int ii, jj;

  resetMatrix(localA, row_a, col_a);
  
  for(int my_i = 0; my_i < row_a; my_i++){
      //get global index from local index
      ii = (((my_i/Mb) * nprow) + myrow)*Mb + my_i%Mb; 
      for(int my_j = 0; my_j < col_a; my_j++){
      	jj = (((my_j/Nb) * npcol) + mycol)*Nb + my_j%Nb; 
	printf("local: (%d,%d) -> global: (%d,%d)\n",my_i, my_j, ii, jj);
	localA[my_i][my_j] = A_glob[ii][jj]; 
      }
  }

  //printf("local A in Pid: %d\n", myrank);
  //printMatrix(localA, row_a, col_a);
  //
  /*
  resetMatrix(localB, row_b, col_b);
  for(int my_i = 0; my_i < row_b; my_i++){
      //get global index from local index
      ii = (((my_i/Mb) * nprow) + myrow)*Mb + my_i%Mb; 
      for(int my_j = 0; my_j < col_b; my_j++){
      	jj = (((my_j/Nb) * npcol) + mycol)*Nb + my_j%Nb; 
	//printf("local: (%d,%d) -> global: (%d,%d)\n",my_i, my_j, ii, jj);
	localB[my_i][my_j] = B_glob[ii][jj]; 
      }
  }
  resetMatrix(localC, row_c, col_c);
  */
  //
  double alpha = 1.0;
  double beta = 0.0;
  



  /* Prepare array descriptors for ScaLAPACK */
  int desc_a[9], desc_b[9], desc_c[9];
  int info;
  //descinit_( desc_a,  &M, &K, &Mb, &Nb, &iZERO, &iZERO, &context, &row_a, &info);
  //if(info != 0) {
  //      printf("Error in descinit, info = %d\n", info);
  //}
  //descinit_( desc_b,  &K, &N, &Mb, &Nb, &iZERO, &iZERO, &context, &row_b, &info);
  //descinit_( desc_c,  &M, &N, &Mb, &Nb, &iZERO, &iZERO, &context, &row_c, &info);
  //printf("Completed distributing the matrices.. \n");
  //std::cout<<desc_a[0]<<std::endl;
  //descinit_(desc_a, &M, &K, &Mb, &Nb, &a, &a, &context, &row_a, &info);
  //descinit_(desc_b, &K, &N, &Mb, &Nb, 0, 0, &context, &row_b, &info);
  //descinit_(desc_c, &M, &N, &Mb, &Nb, 0, 0, &context, &row_c, &info);
  //int ia = myrow;
  //int ja = mycol;
  //int ib = myrow;
  //int jb = mycol;
  //int ic = myrow;
  //int jc = mycol;

  //pdgemm_("N", "N", &M, &N, &K, &alpha, a,  &row_a, &myrow,&mycol, desc_a, b, &row_a, &myrow, &mycol, desc_b, &beta, c, &row_c, &myrow,&mycol, desc_c);

  //pdgemm_("N", "N", &M, &N, &K, &alpha, a, &myrow, &mycol, desc_a, b, &myrow, &mycol, desc_b, &beta, c, &myrow, &mycol, desc_c);

  //pdgemm_("N", "N", &M, &N, &K, &alpha, a, &row_a, &myrow, &mycol, b, &row_b, &myrow, &mycol, &beta, c, &row_c, &myrow, &mycol);



  //delete[] localA;
  //delete[] localB;
  //delete[] localC;

  Cblacs_gridexit(context);
  MPI_Finalize();

  return 0;
}

