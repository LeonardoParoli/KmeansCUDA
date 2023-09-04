
#ifndef KMEANS_KMEANSCUDA_CUH
#define KMEANS_KMEANSCUDA_CUH

#include "../Point.h"
#include "../Kluster.h"
#include <cuda_runtime.h>

Points transformAoStoSoA(Point* points, int numPoints,int gridSize, int blockSize);
void calculateMaxSSE_SOA( Points soaPoints,Point* selectedCentroids, int numPoints, int numClusters, int numBlocks, int threadsPerBlock, double& maxSSE);
Kluster* kmeansCycle_SOA(Points soaPoints,int numPoints, Point* selectedCentroids, int numClusters, float maxSSE, bool printConsole);

__global__ void transformAoaToSoa(const Point* d_aoaPoints, int numPoints, Points d_soaPoints);
__global__ void CUDAcalculateMaxSSE_SOA(Points d_soaPoints, Point* d_currentCentroids, double* d_maxSSE, int numPoints, int numClusters);
__global__ void assignPointsToClusters_SOA(Points d_soaPoints, int numPoints, int numClusters, const Point* d_currentCentroids, int* d_assignments);
    //__global__ void assignPointsToClusters_SOA(const double* d_x, const double* d_y, const double* d_z, int numPoints, int numClusters, const Point* d_currentCentroids, int* d_assignments);
//__global__ void assignPointsToClusters_SOA(Points soaPoints, int numPoints, int numClusters, Point* d_currentCentroids, int* d_assignment);
__global__ void calculateSSEKernel_SOA(Points soaPoints, int numPoints, const int* assignments, Point* newCentroids, double* currentSSE);
__global__ void calculateNewCentroidsKernel_SOA(int numPoints, Points soaPoints, const int* assignments, Point* newCentroids, int* clusterSizes);
__global__ void calculateFinalCentroidsKernel(Point* newCentroids, const int* clusterSizes, int numClusters);

void kickstartGPUCUDA();
__global__ void vectorAdd(int* a, int* b, int* c, int n);

/////////////// AoS methods -> improved by SoA methods ^^^^^
void calculateMaxSSE( Point* points,Point* selectedCentroids, int numPoints, int numClusters, int gridSize, int blockSize, double& maxSSE);
Kluster* kmeansCycle(Point* points,int numPoints, Point* selectedCentroids, int numClusters, float maxSSE, bool printConsole);

__global__ void CUDAcalculateMaxSSE(Point* d_points, Point* d_currentCentroids, double* d_maxSSE, int numPoints, int numClusters);
__global__ void assignPointsToClusters(Point* points, int numPoints, int numClusters, Point* d_currentCentroids, int* d_assignment);
__global__ void calculateSSEKernel(Point* points, int numPoints, int* assignments, Point* newCentroids, double* currentSSE);
__global__ void calculateNewCentroidsKernel(int numPoints, Point* points, const int* assignments, Point* newCentroids, int* clusterSizes);


#endif //KMEANS_KMEANSCUDA_CUH