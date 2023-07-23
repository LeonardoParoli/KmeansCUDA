
#ifndef KMEANS_KMEANSCUDA_CUH
#define KMEANS_KMEANSCUDA_CUH

#include "../Point.h"
#include "../Kluster.h"
#include <cuda_runtime.h>

Kluster* kmeansCycle(Point* points,int numPoints, Point* selectedCentroids, int numClusters, float maxSSE, bool printConsole);
void calculateMaxSSE( Point* points,Point* selectedCentroids, int numPoints, int numClusters, int numBlocks, int threadsPerBlock, float& maxSSE);

__global__ void CUDAcalculateMaxSSE(Point* d_points, Point* d_currentCentroids, float* d_maxSSE, int numPoints, int numClusters);
__global__ void assignPointsToClusters(Point* points, int numPoints, int numClusters, Point* d_currentCentroids, int* d_assignment);

#endif //KMEANS_KMEANSCUDA_CUH