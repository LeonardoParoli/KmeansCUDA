
#ifndef KMEANS_KMEANSCUDA_CUH
#define KMEANS_KMEANSCUDA_CUH

#include "../Point.h"
#include <cuda_runtime.h>

void kmeansIteration(Point* d_points, Point* points,int numPoints, Point* d_currentCentroids, Point* newCentroids, int numClusters);
void calculateMaxSSE(Point* d_points, Point* points, Point* d_currentCentroids, Point* selectedCentroids, float* d_maxSSE, int numPoints, int numClusters, int numBlocks, int threadsPerBlock, float& maxSSE);

__global__ void CUDAcalculateMaxSSE(Point* d_points, Point* d_currentCentroids, float* d_maxSSE, int numPoints, int numClusters);
#endif //KMEANS_KMEANSCUDA_CUH