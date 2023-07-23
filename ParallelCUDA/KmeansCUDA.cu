#include "KmeansCUDA.cuh"
#include <iostream>
#include <random>
#include "../Point.h"


static void CheckCudaErrorAux(const char*, unsigned, const char*, cudaError_t);

static void CheckCudaErrorAux(const char* file, unsigned line, const char* statement, cudaError_t err) {
    if (err == cudaSuccess)
        return;
    std::cerr << statement << " returned " << cudaGetErrorString(err) << "(" << err << ") at " << file << ":" << line << std::endl;
    exit(1);
}

#define CUDA_CHECK_ERROR(value) CheckCudaErrorAux(__FILE__,__LINE__, #value, value)
__host__ void calculateMaxSSE(Point*points, Point* selectedCentroids, int numPoints, int numClusters, int gridSize, int blockSize, float& maxSSE) {
    Point* d_points;
    Point* d_currentCentroids;
    float* d_maxSSE;
    CUDA_CHECK_ERROR(cudaMalloc((void**)&d_points, numPoints * sizeof(Point)));
    CUDA_CHECK_ERROR(cudaMalloc((void**)&d_currentCentroids, numClusters * sizeof(Point)));
    CUDA_CHECK_ERROR(cudaMalloc((void**)&d_maxSSE, sizeof(double)));
    CUDA_CHECK_ERROR(cudaMemcpy(d_points, points, numPoints * sizeof(Point), cudaMemcpyHostToDevice));
    CUDA_CHECK_ERROR(cudaMemcpy(d_currentCentroids, selectedCentroids, numClusters * sizeof(Point), cudaMemcpyHostToDevice));

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

__host__ Kluster* kmeansCycle(Point* points,int numPoints,Point* selectedCentroids, int numClusters, float maxSSE, bool printConsole) {
    //CUDA parameters:
    int blockSize=256;
    int gridSize = (numPoints + blockSize - 1) / blockSize;

    Point* d_points;
    Point* d_currentCentroids;
    auto *newCentroids = new Point[numClusters];
    for(int i = 0; i < numClusters; i++){
        newCentroids[i] = selectedCentroids[i];
    }
    CUDA_CHECK_ERROR(cudaMalloc((void **) &d_points, numPoints * sizeof(Point)));
    CUDA_CHECK_ERROR(cudaMalloc((void **) &d_currentCentroids, numClusters * sizeof(Point)));
    CUDA_CHECK_ERROR(cudaMemcpy(d_points, points, numPoints * sizeof(Point), cudaMemcpyHostToDevice));

    float currentSSE=maxSSE;
    float previousSSE = 1e20;
    int iteration = 0;
    Point filler = {-1,-1,-1};
    auto * finalClusters = new Kluster[numClusters];
    for(int j=0; j<numClusters; j++){
        finalClusters[j] = Kluster();
    }
    int *assignments = new int[numPoints];
    while ((previousSSE - currentSSE) >= 0.01 && iteration < 10000) {
        previousSSE = currentSSE;
        CUDA_CHECK_ERROR(cudaMemcpy(d_currentCentroids, newCentroids, numClusters * sizeof(Point), cudaMemcpyHostToDevice));
        //clear clusters
        delete[] assignments;
        assignments = new int[numPoints];

        //assigning points to clusters
        int *d_assignments;
        CUDA_CHECK_ERROR(cudaMalloc((void **) &d_assignments, numPoints * sizeof(int)));
        assignPointsToClusters<<<gridSize, blockSize>>>(d_points, numPoints, numClusters, d_currentCentroids, d_assignments);
        CUDA_CHECK_ERROR(cudaMemcpy(assignments, d_assignments, numPoints * sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK_ERROR(cudaFree(d_assignments));

        //Update centroids
        int *clusterSizes = new int[numClusters];
        for(int i=0; i < numClusters; i++){
            clusterSizes[i]=0;
        }
        for(int j=0; j < numPoints; j++){
            clusterSizes[assignments[j]]++;
        }
        for(int i = 0; i < numClusters; i++) {
            if (clusterSizes[i] == 0) {
                std::random_device rd;
                std::mt19937 rng(rd());
                std::uniform_real_distribution<double> dist(0.0, 1000.0);
                newCentroids[i].x = dist(rng);
                newCentroids[i].y = dist(rng);
                newCentroids[i].z = dist(rng);
            }else{
                newCentroids[i] = {0.0, 0.0, 0.0};
            }
        }
        for(int i = 0; i < numPoints; i++){
            Point point = points[i];
            int assignment = assignments[i];
            newCentroids[assignment].x += point.x;
            newCentroids[assignment].y += point.y;
            newCentroids[assignment].z += point.z;
        }
        for(int i = 0; i < numClusters; i++){
            newCentroids[i].x = newCentroids[i].x / clusterSizes[i];
            newCentroids[i].y = newCentroids[i].y / clusterSizes[i];
            newCentroids[i].z = newCentroids[i].z / clusterSizes[i];
        }

        //Update currentSSE
        currentSSE = 0.0;
        for(int i = 0; i < numPoints; i++){
            int assignment = assignments[i];
            Point point = points[i];
            Point assignedCentroid = newCentroids[assignment];
            double distance = 0;
            distance += pow(point.x - assignedCentroid.x, 2);
            distance += pow(point.y - assignedCentroid.y, 2);
            distance += pow(point.z - assignedCentroid.z, 2);
            currentSSE += distance;
        }
        currentSSE = currentSSE/numPoints;
        if(printConsole){
            std::cout <<"Current SSE = " << currentSSE << "" << std::endl;
        }
        //update iteration
        iteration++;
    }
    CUDA_CHECK_ERROR(cudaFree(d_points));
    CUDA_CHECK_ERROR(cudaFree(d_currentCentroids));

    //saving iteration on Klusters
    for(int i = 0; i < numPoints; i++){
        finalClusters[assignments[i]].addPoint(&points[i]);
    }
    for (int i = 0; i < numClusters; i++) {
        finalClusters[i].setCentroid(&newCentroids[i]);
    }
    delete[] assignments;

    return finalClusters;
}

__global__ void assignPointsToClusters(Point* d_points, int numPoints, int numClusters, Point* d_currentCentroids, int* d_assignments) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = tid; i < numPoints; i += stride) {
        Point point = d_points[i];
        double minDistance = DBL_MAX;
        int nearestCluster = 0;

        for (int j = 0; j < numClusters; ++j) {
            double distance = 0.0;
            distance += pow(point.x - d_currentCentroids[j].x, 2);
            distance += pow(point.y - d_currentCentroids[j].y, 2);
            distance += pow(point.z - d_currentCentroids[j].z, 2);
            distance = sqrt(distance);

            if (distance < minDistance) {
                minDistance = distance;
                nearestCluster = j;
            }
        }

        d_assignments[i] = nearestCluster;
    }
}