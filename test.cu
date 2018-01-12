#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <sys/time.h>
#include <string.h>
#include <cuda_runtime.h>

#define CHECK(call)\
{\
	const cudaError_t error = call;\
	if (error != cudaSuccess)\
	{\
		printf("Error: %s:%d, ", __FILE__, __LINE__);\
		printf("code:%d, reason: %s\n", error, cudaGetErrorString(error));\
		exit(1);\
	}\
}

// FUNCTIONS FOR SETUP
int numberCells(double, double *);
void setVector(int, double, double *, double *);
void setObst(double *);
void setGoal(double *, double *, double *);
void setInitialValue(double *, double *, double *);
void conditionValue(double *, double *, double *, int, int, int);
void setInitialPolicy(double *, double *, double *);
void conditionPolicy(double *, double *, double *, int, int, int);

// FUNCTIONS FOR VALUE ITERATION
__global__ void valueIteration(double *, double *, double *, double *, double *);
__device__ void conditionR(int, int, int, double *, double *);
__device__ void conditionTheta(int, int, int, double *, double *);
__device__ void conditionPhi(int, int, int, double *, double *);
__device__ void computeTotalCost(double *, double *);
__device__ double computeNewValue(double *);
__device__ double computeNewPolicy(double *);

// FUNCTIONS FOR ANALYSIS
double cpuSecond(void);

// DEFINE GLOBAL PARAMETERS IN CPU
int nr, ntheta, nphi;
double perr;
double gamma1;
double vGoal, vObst, vMove;
double vInitial;
int numActions=7;

// DEFINE GLOBAL PARAMETERS IN GPU
__constant__ int d_nr, d_ntheta, d_nphi;
__constant__ double d_perr;
__constant__ double d_gamma1;
__constant__ double d_vGoal, d_vObst, d_vMove;
__constant__ double d_vInitial;
__constant__ int d_numActions;

/*
__global__
void helloFromGPU()
{
        printf("blockIdx.x=%d,threadIdx.x=%d,blockDim.x=%d\n",blockIdx.x,threadIdx.x,blockDim.x);
}
*/

int main(int argc, char **argv){
	double iStart = cpuSecond();

	// DEFINE PARAMETERS
	double dr, dtheta, dphi;
	double rdim[2], thetadim[2], phidim[2];
	double *rVec, *thetaVec, *phiVec;
	// - minimum grid resolution for r, theta, phi
	dr = 1.0, dtheta = 1.0, dphi = 1.0;
	// - dimensions of the state space
	rdim[0] = 0.0, rdim[1] = 10.0;
	thetadim[0] = 0.0, thetadim[1] = 360.0;
	phidim[0] = 0.0, phidim[1] = 360.0;
	// - number of grid cells
	nr = numberCells(dr, rdim);
	ntheta = numberCells(dtheta, thetadim);
	nphi = numberCells(dphi, phidim);
	// - vectors for r, theta, phi
	rVec = (double *)malloc(sizeof(double)*nr);
	thetaVec = (double *)malloc(sizeof(double)*ntheta);
	phiVec = (double *)malloc(sizeof(double)*nphi);
	setVector(nr, dr, rdim, rVec);
	setVector(ntheta, dtheta, thetadim, thetaVec);
	setVector(nphi, dphi, phidim, phiVec);
	// - probability of going the wrong way
	perr = 0.0;
	// attenuation rate
	gamma1 = 1.0;
	// - value of goal, collision, movement
	vGoal = 100.0;
	vObst = -100.0;
	vMove = -1.0;
	// initial guess at all values
	vInitial = 0.0;

	// DEFINE OBSTACLE AND GOAL LOCATIONS
	double *isobst, *isgoal;
	isobst = (double *)calloc(nr*ntheta*nphi, sizeof(double));
	isgoal = (double *)calloc(nr*ntheta*nphi, sizeof(double));
	setObst(isobst);
	setGoal(thetaVec, phiVec, isgoal);

	// DEFINE OBSTACLE AND GOAL LOCATIONS IN GPU
	double *d_isobst, *d_isgoal;
	CHECK(cudaMalloc((double**)&d_isobst, nr*ntheta*nphi*sizeof(double)));
	CHECK(cudaMalloc((double**)&d_isgoal, nr*ntheta*nphi*sizeof(double)));
	CHECK(cudaMemcpy(d_isobst, isobst, nr*ntheta*nphi*sizeof(double), cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(d_isgoal, isgoal, nr*ntheta*nphi*sizeof(double), cudaMemcpyHostToDevice));

	// DEFINE INITIAL GUESS AT VALUE AND POLICY
	double *J;
	double *U;
	J = (double *)calloc(nr*ntheta*nphi, sizeof(double));
	U = (double *)calloc(nr*ntheta*nphi, sizeof(double));
	setInitialValue(isobst, isgoal, J);
	setInitialPolicy(isobst, isgoal, U);

	// DO VALUE ITERATION
	double *Jprev;
	double *Uprev;
	Jprev = (double *)calloc(nr*ntheta*nphi, sizeof(double));
	Uprev = (double *)calloc(nr*ntheta*nphi, sizeof(double));
	
	// TRANSFER VARIABLE DATA FROM HOST TO DEVICE
	CHECK(cudaMemcpyToSymbol(d_nr, &nr, sizeof(int), 0, cudaMemcpyHostToDevice));
	CHECK(cudaMemcpyToSymbol(d_ntheta, &ntheta, sizeof(int), 0, cudaMemcpyHostToDevice));
	CHECK(cudaMemcpyToSymbol(d_nphi, &nphi, sizeof(int), 0, cudaMemcpyHostToDevice));
	CHECK(cudaMemcpyToSymbol(d_perr, &perr, sizeof(double), 0, cudaMemcpyHostToDevice));
	CHECK(cudaMemcpyToSymbol(d_gamma1, &gamma1, sizeof(double), 0, cudaMemcpyHostToDevice));
	CHECK(cudaMemcpyToSymbol(d_vGoal, &vGoal, sizeof(double), 0, cudaMemcpyHostToDevice));
	CHECK(cudaMemcpyToSymbol(d_vObst, &vObst, sizeof(double), 0, cudaMemcpyHostToDevice));
	CHECK(cudaMemcpyToSymbol(d_vMove, &vMove, sizeof(double), 0, cudaMemcpyHostToDevice));
	CHECK(cudaMemcpyToSymbol(d_vInitial, &vInitial, sizeof(double), 0, cudaMemcpyHostToDevice));
	CHECK(cudaMemcpyToSymbol(d_numActions, &numActions, sizeof(int), 0, cudaMemcpyHostToDevice));
	float error=1;
	int t=1;
	while(error!=0){
		printf("Iteration %d\n", t);

		// Iterate over all states.
		memcpy(Jprev, J, sizeof(double)*nr*ntheta*nphi);
		memcpy(Uprev, U, sizeof(double)*nr*ntheta*nphi);

		// allocate memory at device
		double  *d_J, *d_U, *d_Jprev, *d_Uprev;
		CHECK(cudaMalloc((double**)&d_J, nr*ntheta*nphi*sizeof(double)));
		CHECK(cudaMalloc((double**)&d_U, nr*ntheta*nphi*sizeof(double)));
		CHECK(cudaMalloc((double**)&d_Jprev, nr*ntheta*nphi*sizeof(double)));
		CHECK(cudaMalloc((double**)&d_Uprev, nr*ntheta*nphi*sizeof(double)));

		// transfer data from host to device
		CHECK(cudaMemcpy(d_J, J, nr*ntheta*nphi*sizeof(double), cudaMemcpyHostToDevice));
		CHECK(cudaMemcpy(d_U, U, nr*ntheta*nphi*sizeof(double), cudaMemcpyHostToDevice));
		CHECK(cudaMemcpy(d_Jprev, Jprev, nr*ntheta*nphi*sizeof(double), cudaMemcpyHostToDevice));
		CHECK(cudaMemcpy(d_Uprev, Uprev, nr*ntheta*nphi*sizeof(double), cudaMemcpyHostToDevice));

		// configure number of threads and blocks
		//printf("nr = %d ntheta = %d nphi = %d\n",nr,ntheta, nphi);
		dim3 nThreads(1,1,1);
		dim3 nBlocks((nr+nThreads.x-1)/nThreads.x,(ntheta+nThreads.y-1)/nThreads.y,(nphi+nThreads.z-1)/nThreads.z);
		//printf("nBlocks.x=%d nBlocks.y=%d nBlocks.z=%d\n", nBlocks.x,nBlocks.y,nBlocks.z);	
		
		// call kernel
		//helloFromGPU<<<nBlocks,nThreads>>>();
		valueIteration<<<nBlocks, nThreads>>>(d_isobst, d_isgoal, d_J, d_U, d_Jprev);
		CHECK(cudaDeviceSynchronize());

		// copy result from device to host
		CHECK(cudaMemcpy(J, d_J, nr*ntheta*nphi*sizeof(double), cudaMemcpyDeviceToHost));
		CHECK(cudaMemcpy(U, d_U, nr*ntheta*nphi*sizeof(double), cudaMemcpyDeviceToHost));

		CHECK(cudaFree(d_J));
		CHECK(cudaFree(d_U));
		CHECK(cudaFree(d_Jprev));
		CHECK(cudaFree(d_Uprev));
		
		error=0;
		for(int x=0; x<nr*ntheta*nphi; x++){
			//printf("%2d J=%3.1f Jprev= %3.1f U=%2f\n", x, J[x], Jprev[x],U[x]);
			error+=(J[x]-Jprev[x]);
		}
		t+=1;
		printf("\n");
	}

	// FREE USED MEMORY IN CPU
	free(rVec);
	free(thetaVec);
	free(phiVec);
	free(isobst);
	free(isgoal);
	free(J);
	free(U);
	free(Jprev);
	free(Uprev);
	
	// FREE USED MEMORY IN GPU
	CHECK(cudaFree(d_isobst));
	CHECK(cudaFree(d_isgoal));

	double iElaps = cpuSecond()-iStart;
	printf("Time elapsed on GPU = %f ms\n", iElaps*1000.0f);

	return(0);
}

/*--------------- FUNCTIONS FOR SETUP ----------------*/

int numberCells(double d, double *dim){
	int n = 0;
	double diff;
	diff = dim[1]-dim[0];

	if(d<1 || d>diff){
		printf("value of resolution or dimension is invalid.\n");
	}
	else{
		n = floorf(diff/d+1.0);
	}

	return n;
}

void setVector(int n, double d, double *dim, double *Vec){
	double value;
	value = dim[0];

	for(int i=0; i<n; i++){
		Vec[i] = value;
		value += d;
	}
}

void setObst(double *isobst){
	for(int j=0; j<ntheta; j++){
		for(int k=0;k<nphi; k++){
			isobst[nr*ntheta*k+(nr-1)*ntheta+j] = 1;
		}
	}
}

void setGoal(double *thetaVec, double *phiVec, double *isgoal){
	for(int j=0; j<ntheta; j++){
		for(int k=0; k<nphi; k++){
			if(thetaVec[j]==phiVec[k])
				isgoal[nr*ntheta*k+j] = 1;
		}
	}
}

void setInitialValue(double *isobst, double *isgoal, double *J){
	for(int i=0; i<nr; i++){
		for(int j=0; j<ntheta; j++){
			for(int k=0; k<nphi; k++){
				conditionValue(isobst, isgoal, J, i, j, k);
			}
		}
	}
}

void conditionValue(double *isobst, double *isgoal, double *J, int i, int j, int k){
	if(isobst[nr*ntheta*k+ntheta*i+j]){
		J[nr*ntheta*k+ntheta*i+j] = vObst;
	}
	else if(isgoal[nr*ntheta*k+ntheta*i+j]){
		J[nr*ntheta*k+ntheta*i+j] = vGoal;
	}
	else{
		J[nr*ntheta*k+ntheta*i+j] = vInitial;
	}
}

void setInitialPolicy(double *isobst, double *isgoal, double *U){
	srand((unsigned)time(NULL));

	for(int i=0; i<nr; i++){
		for(int j=0; j<ntheta; j++){
			for(int k=0; k<nphi; k++){
				conditionPolicy(isobst, isgoal, U, i, j, k);
			}
		}
	}
}

void conditionPolicy(double *isobst, double *isgoal, double *U, int i, int j, int k){
	if(isobst[nr*ntheta*k+ntheta*i+j]){
		U[nr*ntheta*k+ntheta*i+j] = -1;
	}
	else if(isgoal[nr*ntheta*k+ntheta*i+j]){
		U[nr*ntheta*k+ntheta*i+j] = -1;
	}
	else{
		double r = rand() % 7;
		U[nr*ntheta*k+ntheta*i+j] = r;
	}
}

/*--------------- FUNCTIONS FOR VALUE ITERATION ----------------*/

__global__ void valueIteration(double *isobst, double *isgoal, double *J, double *U, double *Jprev){
	int i=blockIdx.x*blockDim.x+threadIdx.x;
	int j=blockIdx.y*blockDim.y+threadIdx.y;
	int k=blockIdx.z*blockDim.z+threadIdx.z;
	//printf("i=%d j=%d k=%d\n", blockIdx.x,j,k);
	double *tempCost, *totalCost;
	tempCost = (double*)malloc(d_numActions*sizeof(double));
	totalCost = (double*)malloc(d_numActions*sizeof(double));

	if(i<d_nr && j<d_ntheta && k<d_nphi){
		if(!isobst[d_nr*d_ntheta*k+d_ntheta*i+j] && !isgoal[d_nr*d_ntheta*k+d_ntheta*i+j]){
			tempCost[0]=Jprev[d_nr*d_ntheta*k+d_ntheta*i+j];
			// condition of r
			conditionR(i, j, k, tempCost, Jprev);

			// Compute the total expected cost for each of the possible actions.
			computeTotalCost(tempCost, totalCost);

			// Compute the new exptected cost-to-go, by taking the maximum over
			// possible actions.
			J[d_nr*d_ntheta*k+d_ntheta*i+j] = computeNewValue(totalCost);
			U[d_nr*d_ntheta*k+d_ntheta*i+j] = computeNewPolicy(totalCost);
		}
	}
	
	free(tempCost);
	free(totalCost);

	__syncthreads();
}

__device__ void conditionR(int i, int j, int k, double *tempCost, double *Jprev){
	if(i==0){
		tempCost[1] = Jprev[d_nr*d_ntheta*k+d_ntheta*(i+1)+j];
		tempCost[2] = Jprev[d_nr*d_ntheta*k+d_ntheta*i+j];
	}
	else{
		tempCost[1] = Jprev[d_nr*d_ntheta*k+d_ntheta*(i+1)+j];
		tempCost[2] = Jprev[d_nr*d_ntheta*k+d_ntheta*(i-1)+j];
	}
	conditionTheta(i, j, k, tempCost, Jprev);
}

__device__ void conditionTheta(int i, int j, int k, double *tempCost, double *Jprev){
	if(j==0){
		tempCost[3] = Jprev[d_nr*d_ntheta*k+d_ntheta*i+(j+1)];
		tempCost[4] = Jprev[d_nr*d_ntheta*k+d_ntheta*i+(d_ntheta-1)];
	}
	else if(j==d_ntheta-1){
		tempCost[3] = Jprev[d_nr*d_ntheta*k+d_ntheta*i];
		tempCost[4] = Jprev[d_nr*d_ntheta*k+d_ntheta*i+(j-1)];
	}
	else{
		tempCost[3] = Jprev[d_nr*d_ntheta*k+d_ntheta*i+(j+1)];
		tempCost[4] = Jprev[d_nr*d_ntheta*k+d_ntheta*i+(j-1)];
	}
	conditionPhi(i, j, k, tempCost, Jprev);
}

__device__ void conditionPhi(int i, int j, int k, double *tempCost, double *Jprev){
	if(k==0){
		tempCost[5] = Jprev[d_nr*d_ntheta*(k+1)+d_ntheta*i+j];
		tempCost[6] = Jprev[d_nr*d_ntheta*(d_nphi-1)+d_ntheta*i+j];
	}
	else if(k==d_nphi-1){
		tempCost[5] = Jprev[d_ntheta*i+j];
		tempCost[6] = Jprev[d_nr*d_ntheta*(k-1)+d_ntheta*i+j];
	}
	else{
		tempCost[5] = Jprev[d_nr*d_ntheta*(k+1)+d_ntheta*i+j];
		tempCost[6] = Jprev[d_nr*d_ntheta*(k-1)+d_ntheta*i+j];
	}
}

__device__ void computeTotalCost(double *tempCost, double *totalCost){
	double tempCostTotal=0;
	
	for(int n=0; n<d_numActions; n++){
		tempCostTotal+=tempCost[n];
	}
	for(int n=0; n<d_numActions; n++){
		totalCost[n]=d_vMove+d_gamma1*((1-d_perr)*tempCost[n]+(d_perr/6)*(tempCostTotal-tempCost[n]));
	}
}

__device__ double computeNewValue(double *totalCost){
	double max;
	max = totalCost[0];

	for(int n=0; n<d_numActions; n++){
		if(totalCost[n]>max)
			max=totalCost[n];
	}

	return max;
}

__device__ double computeNewPolicy(double *totalCost){
	double max;
	double idx;
	max = totalCost[0];

	for(int n=0; n<d_numActions; n++){
		if(totalCost[n]>max){
			max=totalCost[n];
			idx=n;
		}

	}

	return idx;
}

/*-------------- FUNCTION FOR ANALYSIS --------------*/
double cpuSecond(void){
	struct timeval tp;
	gettimeofday(&tp, NULL);
	return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}
