#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <cuda_runtime.h>
#include "vector.h"
#include "config.h"
#include "planets.h"
#include "compute.h"

// represents the objects in the system.  Global variables
vector3 *hVel, *d_hvel;
vector3 *hPos, *d_hpos;
double *mass, *d_mass; //dmass: to be passed onto the device memory.


// Initialize accels array using CUDA kernel function.
__global__ void initializeAccels(vector3 **accels, vector3 *values, int numEntities) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < numEntities) {
        accels[idx] = &values[idx*numEntities];
    }
}


// CUDA kernel to compute pairwise accelerations
__global__ void computeAccels(vector3** accels, vector3* hPos, double* mass) {
    
    int i = blockDim.y * blockIdx.y + threadIdx.y;
    int j = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < NUMENTITIES && j < NUMENTITIES) {
        if (i == j) {
            FILL_VECTOR(accels[i][j], 0, 0, 0);
        } else {
            vector3 distance;
            for (int l = 0; l < 3; l++) {
                distance[l] = hPos[i][l] - hPos[j][l];
            }
            double magnitude_sq = distance[0] * distance[0] + distance[1] * distance[1] + distance[2] * distance[2];
            double magnitude = sqrt(magnitude_sq);
            double accelmag = -1 * GRAV_CONSTANT * mass[j] / magnitude_sq;
            FILL_VECTOR(accels[i][j], accelmag * distance[0] / magnitude, accelmag * distance[1] / magnitude, accelmag * distance[2] / magnitude);
        }
    }

}

// CUDA kernel to sum up the rows of the acceleration matrix
__global__ void sumAccels(vector3** accels, vector3* accel_sum) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int k= blockIdx.y;
    if (i < NUMENTITIES) {
	FILL_VECTOR(accel_sum[i], 0, 0, 0);
        for (int j = 0; j < NUMENTITIES; j++) {
                accel_sum[i][k] += accels[i][j][k];
        }
    }
}

// CUDA kernel to update velocity and position
__global__ void updateVelPos(vector3* hVel, vector3* hPos, vector3* accel_sum) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int k= blockIdx.y;
    if (i < NUMENTITIES) {
            hVel[i][k] += accel_sum[i][k] * INTERVAL;
            hPos[i][k] += hVel[i][k] * INTERVAL;
    }
}

//initHostMemory: Create storage for numObjects entities in our system
//Parameters: numObjects: number of objects to allocate
//Returns: None
//Side Effects: Allocates memory in the hVel, hPos, and mass global variables
void initHostMemory(int numObjects)
{
	hVel = (vector3 *)malloc(sizeof(vector3) * numObjects);
	hPos = (vector3 *)malloc(sizeof(vector3) * numObjects);
	mass = (double *)malloc(sizeof(double) * numObjects);
}

//freeHostMemory: Free storage allocated by a previous call to initHostMemory
//Parameters: None
//Returns: None
//Side Effects: Frees the memory allocated to global variables hVel, hPos, and mass.
void freeHostMemory()
{
	free(hVel);
	free(hPos);
	free(mass);
}

//planetFill: Fill the first NUMPLANETS+1 entries of the entity arrays with an estimation
//				of our solar system (Sun+NUMPLANETS)
//Parameters: None
//Returns: None
//Fills the first 8 entries of our system with an estimation of the sun plus our 8 planets.
void planetFill(){
	int i,j;
	double data[][7]={SUN,MERCURY,VENUS,EARTH,MARS,JUPITER,SATURN,URANUS,NEPTUNE};
	for (i=0;i<=NUMPLANETS;i++){
		for (j=0;j<3;j++){
			hPos[i][j]=data[i][j];
			hVel[i][j]=data[i][j+3];
		}
		mass[i]=data[i][6];
	}
}

//randomFill: FIll the rest of the objects in the system randomly starting at some entry in the list
//Parameters: 	start: The index of the first open entry in our system (after planetFill).
//				count: The number of random objects to put into our system
//Returns: None
//Side Effects: Fills count entries in our system starting at index start (0 based)
void randomFill(int start, int count)
{
	int i, j;
	for (i = start; i < start + count; i++)
	{
		for (j = 0; j < 3; j++)
		{
			hVel[i][j] = (double)rand() / RAND_MAX * MAX_DISTANCE * 2 - MAX_DISTANCE;
			hPos[i][j] = (double)rand() / RAND_MAX * MAX_VELOCITY * 2 - MAX_VELOCITY;
			mass[i] = (double)rand() / RAND_MAX * MAX_MASS;
		}
	}
}

//printSystem: Prints out the entire system to the supplied file
//Parameters: 	handle: A handle to an open file with write access to prnt the data to
//Returns: 		none
//Side Effects: Modifies the file handle by writing to it.
void printSystem(FILE* handle){
	int i,j;
	for (i=0;i<NUMENTITIES;i++){
		fprintf(handle,"pos=(");
		for (j=0;j<3;j++){
			fprintf(handle,"%lf,",hPos[i][j]);
		}
		printf("),v=(");
		for (j=0;j<3;j++){
			fprintf(handle,"%lf,",hVel[i][j]);
		}
		fprintf(handle,"),m=%lf\n",mass[i]);
	}

	printf("\n");
}

int main(int argc, char **argv)
{
	clock_t t0=clock();
	int t_now;
	
	srand(1234);
	initHostMemory(NUMENTITIES);
	planetFill();
	randomFill(NUMPLANETS + 1, NUMASTEROIDS);
	//now we have a system.

	#ifdef DEBUG
	printSystem(stdout);
	#endif

	// Allocating memory on device
	cudaMalloc((void**)&d_hvel, sizeof(vector3)*NUMENTITIES);
	cudaMalloc((void**)&d_hpos, sizeof(vector3)*NUMENTITIES);
	cudaMalloc((void**)&d_mass, sizeof(double)*NUMENTITIES);

	// Copying data from host to device
	cudaMemcpy(d_hvel, hVel, sizeof(vector3)*NUMENTITIES, cudaMemcpyHostToDevice);
	cudaMemcpy(d_hpos, hPos, sizeof(vector3)*NUMENTITIES, cudaMemcpyHostToDevice);
	cudaMemcpy(d_mass, mass, sizeof(double)*NUMENTITIES, cudaMemcpyHostToDevice);

	int threads_per_block = 256;
    	int num_blocks = (NUMENTITIES + threads_per_block - 1) / threads_per_block;
	dim3 blockSize(16, 16, 1);
        dim3 gridSize((NUMENTITIES + blockSize.x - 1) / blockSize.x, (NUMENTITIES + blockSize.y - 1) / blockSize.y, 1);

	vector3 *d_values;
	cudaMalloc((void **)&d_values, sizeof(vector3)*NUMENTITIES*NUMENTITIES);
	
	vector3 **d_accels;
	cudaMalloc((void ***)&d_accels, sizeof(vector3*)*NUMENTITIES);
	
	initializeAccels<<<num_blocks, threads_per_block>>>(d_accels, d_values, NUMENTITIES);
	cudaDeviceSynchronize();
	
	vector3* d_accel_sum;
	cudaMalloc((void **)&d_accel_sum, NUMENTITIES * sizeof(vector3));
	dim3 wideblock(num_blocks,3);
	// Call Kernal function for each INTERVAL
	for(t_now = 0; t_now < DURATION; t_now+= INTERVAL){
		computeAccels<<<gridSize, blockSize>>>(d_accels, d_hpos, d_mass);
			
		sumAccels<<<wideblock, threads_per_block>>>(d_accels, d_accel_sum);
		
		updateVelPos<<<wideblock, threads_per_block>>>(d_hvel, d_hpos, d_accel_sum);
				
	}

	// Copying data from device to host
	cudaMemcpy(hVel, d_hvel, sizeof(vector3)*NUMENTITIES, cudaMemcpyDeviceToHost);
	cudaMemcpy(hPos, d_hpos, sizeof(vector3)*NUMENTITIES, cudaMemcpyDeviceToHost);
	
	// Deallocating memory on device
	cudaFree(d_accel_sum);
	cudaFree(d_hvel);
	cudaFree(d_hpos);
	cudaFree(d_mass);
	cudaFree(d_accels);
	cudaFree(d_values);

	clock_t t1=clock()-t0;

	#ifdef DEBUG
	printSystem(stdout);
	#endif

	printf("This took a total time of %f seconds\n",(double)t1/CLOCKS_PER_SEC);

	freeHostMemory();
}

