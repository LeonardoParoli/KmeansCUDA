
#ifndef KMEANS_KMEANSCUDA_CUH
#define KMEANS_KMEANSCUDA_CUH

#include "../Point.h"
#include "../Kluster.h"
#include <cuda_runtime.h>

Kluster* kmeansCycle(Point* points,int numPoints, Point* selectedCentroids, int numClusters, float maxSSE, bool printConsole);
void calculateMaxSSE( Point* points,Point* selectedCentroids, int numPoints, int numClusters, int numBlocks, int threadsPerBlock, double& maxSSE);
void kickstartGPUCUDA();

__global__ void CUDAcalculateMaxSSE(Point* d_points, Point* d_currentCentroids, double* d_maxSSE, int numPoints, int numClusters);
__global__ void assignPointsToClusters(Point* points, int numPoints, int numClusters, Point* d_currentCentroids, int* d_assignment);
__global__ void calculateSSEKernel(Point* points, int numPoints, int* assignments, Point* newCentroids, double* currentSSE);
__global__ void calculateNewCentroidsKernel(int numPoints, Point* points, const int* assignments, Point* newCentroids, int* clusterSizes);
__global__ void calculateFinalCentroidsKernel(Point* newCentroids, const int* clusterSizes, int numClusters);
__global__ void vectorAdd(int* a, int* b, int* c, int n);
#endif //KMEANS_KMEANSCUDA_CUH