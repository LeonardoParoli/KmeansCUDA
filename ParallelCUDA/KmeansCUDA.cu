#include "KmeansCUDA.cuh"
#include <iostream>

static void CheckCudaErrorAux(const char*, unsigned, const char*, cudaError_t);

static void CheckCudaErrorAux(const char* file, unsigned line, const char* statement, cudaError_t err) {
    if (err == cudaSuccess)
        return;
    std::cerr << statement << " returned " << cudaGetErrorString(err) << "(" << err << ") at " << file << ":" << line << std::endl;
    exit(1);
}

#define CUDA_CHECK_ERROR(value) CheckCudaErrorAux(__FILE__,__LINE__, #value, value)
__host__ void calculateMaxSSE(Point* d_points, Point*points, Point* d_currentCentroids, Point* selectedCentroids, float* d_maxSSE, int numPoints, int numClusters, int gridSize, int blockSize, float& maxSSE) {
    CUDA_CHECK_ERROR(cudaMalloc((void**)&d_points, numPoints * sizeof(Point)));
    CUDA_CHECK_ERROR(cudaMalloc((void**)&d_currentCentroids, numClusters * sizeof(Point)));
    CUDA_CHECK_ERROR(cudaMalloc((void**)&d_maxSSE, sizeof(double)));
    CUDA_CHECK_ERROR(cudaMemcpy(d_points, points, numPoints * sizeof(Point), cudaMemcpyHostToDevice));
    CUDA_CHECK_ERROR(cudaMemcpy(d_currentCentroids, selectedCentroids, numClusters * sizeof(Point), cudaMemcpyHostToDevice));
    CUDA_CHECK_ERROR(cudaDeviceSynchronize());

    CUDAcalculateMaxSSE<<<gridSize, blockSize>>>(d_points, d_currentCentroids, d_maxSSE, numPoints, numClusters);

    CUDA_CHECK_ERROR(cudaMemcpy(&maxSSE, d_maxSSE, sizeof(float), cudaMemcpyDeviceToHost));

    CUDA_CHECK_ERROR(cudaFree(d_points));
    CUDA_CHECK_ERROR(cudaFree(d_currentCentroids));
    CUDA_CHECK_ERROR(cudaFree(d_maxSSE));
}

__global__ void CUDAcalculateMaxSSE(Point* d_points, Point* d_currentCentroids, float* d_maxSSE, int numPoints, int numClusters){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int pointsPerThread = (numPoints + blockDim.x - 1) / blockDim.x;
    int startPoint = tid * pointsPerThread;
    int endPoint = min(startPoint + pointsPerThread, numPoints);

    float sse = 0.0f;
    for (int pointIndex = startPoint; pointIndex < endPoint; pointIndex++) {
        Point dataPoint = d_points[pointIndex];
        float minDistance = 0;
        minDistance += pow(dataPoint.x - d_currentCentroids[0].x,2);
        minDistance += pow(dataPoint.y - d_currentCentroids[0].y,2);
        minDistance += pow(dataPoint.z - d_currentCentroids[0].z,2);
        minDistance = sqrt(minDistance);
        for (int clusterId = 1; clusterId < numClusters; clusterId++) {
            float distanceToCentroid = 0;
            distanceToCentroid += pow(dataPoint.x - d_currentCentroids[clusterId].x,2);
            distanceToCentroid += pow(dataPoint.y - d_currentCentroids[clusterId].y,2);
            distanceToCentroid += pow(dataPoint.z - d_currentCentroids[clusterId].z,2);
            distanceToCentroid = sqrt(distanceToCentroid);
            minDistance = fminf(minDistance, distanceToCentroid);
        }

        // Accumulate the SSE for this data point
        sse += minDistance * minDistance;
    }
    atomicAdd(d_maxSSE, sse);
}

__host__ void kmeansIteration(Point* d_points, Point* points,int numPoints, Point* d_currentCentroids, Point* newCentroids, int numClusters) {
    CUDA_CHECK_ERROR(cudaMalloc((void **) &d_points, numPoints * sizeof(Point)));
    CUDA_CHECK_ERROR(cudaMalloc((void **) &d_currentCentroids, numClusters * sizeof(Point)));
    CUDA_CHECK_ERROR(cudaMemcpy(d_points, points, numPoints * sizeof(Point), cudaMemcpyHostToDevice));
    CUDA_CHECK_ERROR(cudaMemcpy(d_currentCentroids, newCentroids, numClusters * sizeof(Point), cudaMemcpyHostToDevice));

    //TODO

    CUDA_CHECK_ERROR(cudaFree(d_points));
    CUDA_CHECK_ERROR(cudaFree(d_currentCentroids));
}





