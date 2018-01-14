#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <sys/time.h>
#include <string.h>

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
int numberCells(float, float *);
void setVector(int, float, float *, float *);
void setObst(float *);
void setGoal(float *, float *, float *);
void setInitialValue(float *, float *, float *);
void conditionValue(float *, float *, float *, int, int, int);
void setInitialPolicy(float *, float *, char *);
void conditionPolicy(float *, float *, char *, int, int, int);

// FUNCTIONS FOR VALUE ITERATION
__global__ void mainOnGPU(float *, float *, float *, char *, float *, char *);
__global__ void valueIteration(float *, float *, float *, char *, float *);
__device__ void conditionR(int, int, int, float *, float *);
__device__ void conditionTheta(int, int, int, float *, float *);
__device__ void conditionPhi(int, int, int, float *, float *);
__device__ void computeTotalCost(float *, float *);
__device__ float computeNewValue(float *);
__device__ float computeNewPolicy(float *);

// FUNCTIONS FOR ANALYSIS
double cpuSecond(void);

// DEFINE GLOBAL PARAMETERS IN CPU
int nr, ntheta, nphi;
float perr;
float gamma1;
float vGoal, vObst, vMove;
float vInitial;
int numActions;

// DEFINE GLOBAL PARAMETERS IN GPU
__constant__ int d_nr, d_ntheta, d_nphi;
__constant__ float d_perr;
__constant__ float d_gamma1;
__constant__ float d_vGoal, d_vObst, d_vMove;
__constant__ float d_vInitial;
__constant__ int d_numActions;

int main(int argc, char **argv)
{
double iStart = cpuSecond();
	
	numActions=7;

	// DEFINE PARAMETERS
	float dr, dtheta, dphi;
	float rdim[2], thetadim[2], phidim[2];
	float *rVec, *thetaVec, *phiVec;
	// - minimum grid resolution for r, theta, phi
	dr = atof(argv[1]), dtheta = atof(argv[2]), dphi = atof(argv[3]);
	// - dimensions of the state space
	rdim[0] = 0.0, rdim[1] = 10.0;
	thetadim[0] = 0.0, thetadim[1] = 360.0;
	phidim[0] = 0.0, phidim[1] = 360.0;
	// - number of grid cells
	nr = numberCells(dr, rdim);
	ntheta = numberCells(dtheta, thetadim);
	nphi = numberCells(dphi, phidim);
	printf("%d,", nr*ntheta*nphi);
	// - vectors for r, theta, phi
	rVec = (float *)malloc(sizeof(float)*nr);
	thetaVec = (float *)malloc(sizeof(float)*ntheta);
	phiVec = (float *)malloc(sizeof(float)*nphi);
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
	float *isobst, *isgoal;
	isobst = (float *)calloc(nr*ntheta*nphi, sizeof(float));
	isgoal = (float *)calloc(nr*ntheta*nphi, sizeof(float));
	setObst(isobst);
	setGoal(thetaVec, phiVec, isgoal);

	// DEFINE OBSTACLE AND GOAL LOCATIONS IN GPU
	float *d_isobst, *d_isgoal;
	CHECK(cudaMalloc((float**)&d_isobst, nr*ntheta*nphi*sizeof(float)));
	CHECK(cudaMalloc((float**)&d_isgoal, nr*ntheta*nphi*sizeof(float)));
	CHECK(cudaMemcpy(d_isobst, isobst, nr*ntheta*nphi*sizeof(float), cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(d_isgoal, isgoal, nr*ntheta*nphi*sizeof(float), cudaMemcpyHostToDevice));

	// DEFINE INITIAL GUESS AT VALUE AND POLICY
	float *J;
	char *U;
	J = (float *)calloc(nr*ntheta*nphi, sizeof(float));
	U = (char *)calloc(nr*ntheta*nphi, sizeof(char));
	setInitialValue(isobst, isgoal, J);
	setInitialPolicy(isobst, isgoal, U);

	// DO VALUE ITERATION
	float *Jprev;
	char *Uprev;
	Jprev = (float *)calloc(nr*ntheta*nphi, sizeof(float));
	Uprev = (char *)calloc(nr*ntheta*nphi, sizeof(char));
	
	// TRANSFER VARIABLE DATA FROM HOST TO DEVICE
	CHECK(cudaMemcpyToSymbol(d_nr, &nr, sizeof(int), 0, cudaMemcpyHostToDevice));
	CHECK(cudaMemcpyToSymbol(d_ntheta, &ntheta, sizeof(int), 0, cudaMemcpyHostToDevice));
	CHECK(cudaMemcpyToSymbol(d_nphi, &nphi, sizeof(int), 0, cudaMemcpyHostToDevice));
	CHECK(cudaMemcpyToSymbol(d_perr, &perr, sizeof(float), 0, cudaMemcpyHostToDevice));
	CHECK(cudaMemcpyToSymbol(d_gamma1, &gamma1, sizeof(float), 0, cudaMemcpyHostToDevice));
	CHECK(cudaMemcpyToSymbol(d_vGoal, &vGoal, sizeof(float), 0, cudaMemcpyHostToDevice));
	CHECK(cudaMemcpyToSymbol(d_vObst, &vObst, sizeof(float), 0, cudaMemcpyHostToDevice));
	CHECK(cudaMemcpyToSymbol(d_vMove, &vMove, sizeof(float), 0, cudaMemcpyHostToDevice));
	CHECK(cudaMemcpyToSymbol(d_vInitial, &vInitial, sizeof(float), 0, cudaMemcpyHostToDevice));
	CHECK(cudaMemcpyToSymbol(d_numActions, &numActions, sizeof(int), 0, cudaMemcpyHostToDevice));
	
	// allocate memory at device
        float  *d_J, *d_Jprev;
	char *d_U, *d_Uprev;
        CHECK(cudaMalloc((float**)&d_J, nr*ntheta*nphi*sizeof(float)));
        CHECK(cudaMalloc((char**)&d_U, nr*ntheta*nphi*sizeof(char)));
        CHECK(cudaMalloc((float**)&d_Jprev, nr*ntheta*nphi*sizeof(float)));
        CHECK(cudaMalloc((char**)&d_Uprev, nr*ntheta*nphi*sizeof(char)));

        // transfer data from host to device
        CHECK(cudaMemcpy(d_J, J, nr*ntheta*nphi*sizeof(float), cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(d_U, U, nr*ntheta*nphi*sizeof(char), cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(d_Jprev, Jprev, nr*ntheta*nphi*sizeof(float), cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(d_Uprev, Uprev, nr*ntheta*nphi*sizeof(char), cudaMemcpyHostToDevice));

	mainOnGPU<<<1,1>>>(d_isobst, d_isgoal, d_J, d_U, d_Jprev, d_Uprev);
	
	// copy result from device to host
        CHECK(cudaMemcpy(J, d_J, nr*ntheta*nphi*sizeof(float), cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(U, d_U, nr*ntheta*nphi*sizeof(char), cudaMemcpyDeviceToHost));

        CHECK(cudaFree(d_J));
        CHECK(cudaFree(d_U));
        CHECK(cudaFree(d_Jprev));
        CHECK(cudaFree(d_Uprev));

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
	printf("%f\n", iElaps*1000.0f);

	return(0);
}

/*--------------- FUNCTIONS FOR SETUP ----------------*/

int numberCells(float d, float *dim)
{
	int n = 0;
	float diff;
	diff = dim[1]-dim[0];

	if(d<0 || d>diff){
		printf("value of resolution or dimension is invalid.\n");
	}
	else{
		n = floorf(diff/d+1.0);
	}

	return n;
}

void setVector(int n, float d, float *dim, float *Vec)
{
	float value;
	value = dim[0];

	for(int i=0; i<n; i++){
		Vec[i] = value;
		value += d;
	}
}

void setObst(float *isobst)
{
	for(int j=0; j<ntheta; j++){
		for(int k=0;k<nphi; k++){
			isobst[nr*ntheta*k+(nr-1)*ntheta+j] = 1;
		}
	}
}

void setGoal(float *thetaVec, float *phiVec, float *isgoal)
{
	for(int j=0; j<ntheta; j++){
		for(int k=0; k<nphi; k++){
			if(thetaVec[j]==phiVec[k])
				isgoal[nr*ntheta*k+j] = 1;
		}
	}
}

void setInitialValue(float *isobst, float *isgoal, float *J)
{
	for(int i=0; i<nr; i++){
		for(int j=0; j<ntheta; j++){
			for(int k=0; k<nphi; k++){
				conditionValue(isobst, isgoal, J, i, j, k);
			}
		}
	}
}

void conditionValue(float *isobst, float *isgoal, float *J, int i, int j, int k)
{
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

void setInitialPolicy(float *isobst, float *isgoal, char *U)
{
	srand((unsigned)time(NULL));

	for(int i=0; i<nr; i++){
		for(int j=0; j<ntheta; j++){
			for(int k=0; k<nphi; k++){
				conditionPolicy(isobst, isgoal, U, i, j, k);
			}
		}
	}
}

void conditionPolicy(float *isobst, float *isgoal, char *U, int i, int j, int k)
{
	if(isobst[nr*ntheta*k+ntheta*i+j]){
		U[nr*ntheta*k+ntheta*i+j] = -1;
	}
	else if(isgoal[nr*ntheta*k+ntheta*i+j]){
		U[nr*ntheta*k+ntheta*i+j] = -1;
	}
	else{
		char r = rand() % numActions;
		U[nr*ntheta*k+ntheta*i+j] = r;
	}
}

/*--------------- FUNCTIONS FOR VALUE ITERATION ----------------*/

__global__ 
void mainOnGPU(float *d_isobst, float *d_isgoal, float *d_J, char *d_U, float *d_Jprev, char *d_Uprev)
{
	dim3 nThreads(2,4,4);
        dim3 nBlocks((d_nr+nThreads.x-1)/nThreads.x,(d_ntheta+nThreads.y-1)/nThreads.y,(d_nphi+nThreads.z-1)/nThreads.z);
	
	float error=1;
        int t=1;
        while(error!=0){
                //printf("Iteration %d\n", t);

                // Iterate over all states.
                memcpy(d_Jprev, d_J, sizeof(float)*d_nr*d_ntheta*d_nphi);
                memcpy(d_Uprev, d_U, sizeof(char)*d_nr*d_ntheta*d_nphi);
    
                // call kernel
                valueIteration<<<nBlocks, nThreads>>>(d_isobst, d_isgoal, d_J, d_U, d_Jprev);
//                CHECK(cudaDeviceSynchronize());
                error=0;
                for(int x=0; x<d_nr*d_ntheta*d_nphi; x++){
                        //printf("%2d d_J=%3.1f d_Jprev= %3.1f d_U=%d\n", x, d_J[x], d_Jprev[x], d_U[x]);
                        error+=(d_J[x]-d_Jprev[x]);
                }
                t+=1;
                //printf("\n");
        }

}
__global__ 
void valueIteration(float *isobst, float *isgoal, float *J, char *U, float *Jprev)
{
	int i=blockIdx.x*blockDim.x+threadIdx.x;
	int j=blockIdx.y*blockDim.y+threadIdx.y;
	int k=blockIdx.z*blockDim.z+threadIdx.z;
	//printf("i=%d j=%d k=%d\n", blockIdx.x,j,k);
	float *tempCost, *totalCost;
	tempCost = (float*)malloc(d_numActions*sizeof(float));
	totalCost = (float*)malloc(d_numActions*sizeof(float));

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

__device__ 
void conditionR(int i, int j, int k, float *tempCost, float *Jprev)
{
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

__device__ 
void conditionTheta(int i, int j, int k, float *tempCost, float *Jprev)
{
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

__device__ 
void conditionPhi(int i, int j, int k, float *tempCost, float *Jprev)
{
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

__device__ 
void computeTotalCost(float *tempCost, float *totalCost)
{
	float tempCostTotal=0;
	
	for(int n=0; n<d_numActions; n++){
		tempCostTotal+=tempCost[n];
	}
	for(int n=0; n<d_numActions; n++){
		totalCost[n]=d_vMove+d_gamma1*((1-d_perr)*tempCost[n]+(d_perr/6)*(tempCostTotal-tempCost[n]));
	}
}

__device__ 
float computeNewValue(float *totalCost)
{
	float max;
	max = totalCost[0];

	for(int n=0; n<d_numActions; n++){
		if(totalCost[n]>max)
			max=totalCost[n];
	}

	return max;
}

__device__ 
float computeNewPolicy(float *totalCost)
{
	float max;
	float idx;
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
double cpuSecond(void)
{
	struct timeval tp;
	gettimeofday(&tp, NULL);
	return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}
