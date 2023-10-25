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
#include <stdexcept>

extern "C" {
    int dgemm_(char *, char *, int *, int *, int *, double *, double *, int *,
              double *, int *, double *, double *, int *);
    
    void dgemv_(char* TRANS, const int* M, const int* N,
               double* alpha, double* A, const int* LDA, double* X,
               const int* INCX, double* beta, double* C, const int* INCY);

    int numroc_(int *n, int *nb, int *iproc, int *srcproc, int *nprocs);
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
      //matrix[i] = (i)%numCols + 1;
      //matrix[i] = (i)%numCols + 1;
      matrix[i] = i + 1;
  }
}


void initVector(double*& vector, int numRows){
  for (int i = 0; i < numRows; i++){
      vector[i] = 1;
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

    if ( nprocs != nprow*npcol ) {
        throw std::invalid_argument( "numrow x numcol must be equal to numprocs" );
    }

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
  //int color_a = my_cart_rank/nprow;
  int color_a = mycol;
  //Determine color based on row
  MPI_Comm_split(world_comm, color_a, my_cart_rank, &row_comm);
  int row_rank, row_size;
  MPI_Comm_rank(row_comm, &row_rank);
  MPI_Comm_size(row_comm, &row_size);

    printf("CART RANK: %d /WORLD RANK: [%d,%d] \t ROW_COMM RANK/SIZE: %d/%d\n",
	        myrank, myrow, mycol, row_rank, row_size);

  /* create col-wise communicators */
  MPI_Comm col_comm;
  int color_b = myrow;
  //Determine color based on col
  MPI_Comm_split(world_comm, color_b, my_cart_rank, &col_comm);
  int col_rank, col_size;
  MPI_Comm_rank(col_comm, &col_rank);
  MPI_Comm_size(col_comm, &col_size);
    //printf("CART RANK/WORLD RANK: [%d,%d] \t COL_COMM RANK/SIZE: %d/%d\n",
	//        myrow, mycol, col_rank, col_size);


  
  /* Read the matrices */
  double *A_glob = new double [M*K];//matrix
  double *X_glob = new double [K];//vector
  double *Y_glob = new double [M];//vector
  //initMatrix(C_glob, M, N);
  if (mpiroot) {
    initMatrix(A_glob, M, K);
    printMatrix(A_glob,M,K);
    //printMatrix(A_glob, M, K);
    //initMatrix(X_glob, K, 1);
    initVector(X_glob, K);
    printMatrix(X_glob,K, 1);
    std::cout<<"====================="<<std::endl;
    //printf("A(%d,%d); blocking: %d,%d\n",M,K, Mb, Nb);
  }
  //printMatrix(A_glob,M,K);
  MPI_Bcast(&M, 1, MPI_INT, 0, world_comm);
  MPI_Bcast(&K, 1, MPI_INT, 0, world_comm);
  MPI_Bcast(&N, 1, MPI_INT, 0, world_comm);

  MPI_Bcast(A_glob, M*K, MPI_DOUBLE, 0, world_comm);
  MPI_Bcast(X_glob, K*1, MPI_DOUBLE, 0, world_comm);

    /* Distribute A in an elemental cyclic distribution*/

    int common = 1, one = 1, zero = 0;
    int row_a = numroc_(&M, &one, &myrow, &zero, &nprow);
    int col_a = numroc_(&K, &one, &mycol, &zero, &npcol);
    double *localA = new double[row_a*col_a];
    int ii, jj;
    for(int i=0; i<M; i++){
        ii = i/nprow;
        for(int j=0; j<K; j++){
            if((i%nprow == myrow) && (j%npcol == mycol)){
                jj = j/npcol;
                localA[ii*col_a + jj] = A_glob[i*K + j];
            }
        }
    }

    for(int i = 0; i < nprocs; ++i) {
    MPI_Barrier(MPI_COMM_WORLD);
    if (myrank == i){
        printf("A in (%d,%d)\n", myrow, mycol);
        printMatrix(localA, row_a, col_a);
        fflush(stdout);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    /* Distribute X in an elemental cyclic distribution*/

    int col_x = numroc_(&K, &one, &my_cart_rank, &zero, &nprocs);
    double *localX = new double[col_x];

    for(int i=0; i<K; i++){
        ii = i/(nprocs);
        jj = i%nprocs;
        //printf("[%d]: %f\n",ii,X_glob[i]);
        //printf("{%d,%d} ,[%d,%d]: %d, %f\n",i/npcol, i%npcol, myrow, mycol, ii,X_glob[i]);
        if((jj/npcol == myrow) && (jj%npcol == mycol)){
            localX[ii] = X_glob[i];
            //printf("{%d,%d} ,[%d,%d]\n",i/npcol, i%npcol, myrow, mycol);
            //printf("[%d,%d]: %d, %f\n",myrow, mycol, ii,X_glob[i]);
            //printf("[%d]: %f\n",ii,X_glob[i]);
        }
    }


    /* create local Y*/

    int row_y = numroc_(&K, &one, &my_cart_rank, &zero, &nprocs);
    double *localY = new double[row_y];
    resetMatrix(localY, row_y, 1);


    /*
    for(int i = 0; i < nprocs; ++i) {
    MPI_Barrier(MPI_COMM_WORLD);
    if (myrank == i){
        printf("X in (%d,%d)\n", myrow, mycol);
        printMatrix(localX, 1, col_x);
        fflush(stdout);
        }
    }
    */

    MPI_Barrier(MPI_COMM_WORLD);
    //std::cout<<"num rows="<<nprow*col_x<<std::endl;
    // Decide the size of the local arrays after MPI_allgather (col_x)
    int mycol_x;
    MPI_Allreduce(&col_x, &mycol_x, 1, MPI_INT, MPI_SUM, row_comm);
    //printf("In (%d,%d), %d size: [%d]\n", myrow, mycol,col_x, mycol_x);
    //MPI_Barrier(row_comm);
    std::cout<<"==== all gather ===="<<std::endl;

    /* Columnwise gather local X */


    double *myworkX = new double[mycol_x];

    MPI_Allgather(&localX[0], col_x, MPI_DOUBLE, &myworkX[0], col_x, MPI_DOUBLE, row_comm);

    for(int i = 0; i < nprocs; ++i) {
    MPI_Barrier(MPI_COMM_WORLD);
    if (myrank == i){
        printf("X in (%d,%d)\n", myrow, mycol);
        //printMatrix(localX, 1, col_x);
        printMatrix(myworkX, 1, mycol_x);
        fflush(stdout);
        }
    }

    /*
    int *recvcounts = new int [nprow];
    for(int i = 0; i < nprow){
        if (i==row_rank){
            recvcounts[i] = col_x;
        }
    }
    int offset = new int [nprow];

    MPI_Allgatherv(&localX[0], recvcounts , MPI_DOUBLE, &myworkX[0], recvcounts,offset, MPI_DOUBLE, row_comm);
    */

    for(int i = 0; i < nprocs; ++i) {
    MPI_Barrier(MPI_COMM_WORLD);
    if (myrank == i){
        printf("Dimension in (%d,%d)\n", myrow, mycol);
        printf("row_a, col_a, mycol_x [%d,%d,%d] \n", row_a, col_a, mycol_x);
        //printMatrix(myworkX, 1, mycol_x);
        fflush(stdout);
        }
    }

    double *myworkY = new double[row_a]; // variable to store Y

    double alpha = 1, beta = 0;

    char transA = 'T', transB = 'N';

    /* matrix vector multiplication */

    //dgemv_(&transA, &row_a, &col_a, &alpha, &*localA, &row_a, &*myworkX, &one, &beta, &*myworkY, &one);

    //char transA = 'T';
    //int row_ax = 2;
    //int col_ax = 4;
    //dgemv_(&transA, &row_ax, &col_ax, &alpha, &*localA, &row_ax, &*myworkX, &one, &beta, &*myworkY, &one);

    dgemv_(&transA, &col_a, &row_a, &alpha, &localA[0], &col_a, &myworkX[0], &one, &beta, &myworkY[0], &one);


    for(int i = 0; i < nprocs; ++i) {
    MPI_Barrier(MPI_COMM_WORLD);
    if (myrank == i){
        printf("Y in (%d,%d)\n", myrow, mycol);
        printMatrix(myworkY, 1, row_a);
        fflush(stdout);
        }
    }

    //printf("=============================\n");

    //double *myY = new double[col_a]; // variable to store Y
    
    /* Now reduce myworkY along rows and we will get the partial contribution of Y  = AX */
    //MPI_Reduce_scatter(&myworkY[0], &myY[0], &mycol_x, MPI_DOUBLE, MPI_SUM, col_comm);
    //int elems = row_a / npcol;
    //printf("num elems in (%d)\n", elems);
    int recvcounts[nprocs];
    for (int i=0; i<nprocs; i++)
        recvcounts[i] = row_a / npcol;

    int elems = row_a / npcol;
    double *myY = new double[elems];
    MPI_Reduce_scatter(&myworkY[0], &myY[0], recvcounts, MPI_DOUBLE, MPI_SUM, col_comm);

    MPI_Barrier(MPI_COMM_WORLD);
    printf("\n");
    for(int i = 0; i < nprocs; ++i) {
    MPI_Barrier(MPI_COMM_WORLD);
    if (myrank == i){
        printf("myY in (%d,%d)\n", myrow, mycol);
        printMatrix(myY, recvcounts[i], 1);
        fflush(stdout);
        }
    MPI_Barrier(MPI_COMM_WORLD);
    }


    delete[] A_glob;
    delete[] X_glob;
    delete[] Y_glob;

    MPI_Comm_free(&world_comm);
    MPI_Finalize();

  return 0;
}

