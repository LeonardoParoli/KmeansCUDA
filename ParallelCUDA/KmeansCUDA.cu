#include "KmeansCUDA.cuh"
#include <iostream>
#include <random>

static void CheckCudaErrorAux(const char*, unsigned, const char*, cudaError_t);
static void CheckCudaErrorAux(const char* file, unsigned line, const char* statement, cudaError_t err) {
    if (err == cudaSuccess)
        return;
    std::cerr << statement << " returned " << cudaGetErrorString(err) << "(" << err << ") at " << file << ":" << line << std::endl;
    exit(1);
}
#define CUDA_CHECK_ERROR(value) CheckCudaErrorAux(__FILE__,__LINE__, #value, value)

__host__ void calculateMaxSSE(Point*points, Point* selectedCentroids, int numPoints, int numClusters, int gridSize, int blockSize, double& maxSSE) {
    Point* d_points;
    Point* d_currentCentroids;
    double* d_maxSSE;
    CUDA_CHECK_ERROR(cudaMalloc((void**)&d_points, numPoints * sizeof(Point)));
    CUDA_CHECK_ERROR(cudaMalloc((void**)&d_currentCentroids, numClusters * sizeof(Point)));
    CUDA_CHECK_ERROR(cudaMalloc((void**)&d_maxSSE, sizeof(double)));
    CUDA_CHECK_ERROR(cudaMemcpy(d_points, points, numPoints * sizeof(Point), cudaMemcpyHostToDevice));
    CUDA_CHECK_ERROR(cudaMemcpy(d_currentCentroids, selectedCentroids, numClusters * sizeof(Point), cudaMemcpyHostToDevice));
    CUDAcalculateMaxSSE<<<gridSize, blockSize>>>(d_points, d_currentCentroids, d_maxSSE, numPoints, numClusters);
    CUDA_CHECK_ERROR(cudaMemcpy(&maxSSE, d_maxSSE, sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK_ERROR(cudaFree(d_points));
    CUDA_CHECK_ERROR(cudaFree(d_currentCentroids));
    CUDA_CHECK_ERROR(cudaFree(d_maxSSE));
}

__global__ void CUDAcalculateMaxSSE(Point* d_points, Point* d_currentCentroids, double* d_maxSSE, int numPoints, int numClusters){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int pointsPerThread = (numPoints + blockDim.x - 1) / blockDim.x;
    int startPoint = tid * pointsPerThread;
    int endPoint = min(startPoint + pointsPerThread, numPoints);

    double sse = 0.0f;
    for (int pointIndex = startPoint; pointIndex < endPoint; pointIndex++) {
        Point dataPoint = d_points[pointIndex];
        double minDistance = 0;
        minDistance += pow(dataPoint.x - d_currentCentroids[0].x,2);
        minDistance += pow(dataPoint.y - d_currentCentroids[0].y,2);
        minDistance += pow(dataPoint.z - d_currentCentroids[0].z,2);
        minDistance = sqrt(minDistance);
        for (int clusterId = 1; clusterId < numClusters; clusterId++) {
            double distanceToCentroid = 0;
            distanceToCentroid += pow(dataPoint.x - d_currentCentroids[clusterId].x,2);
            distanceToCentroid += pow(dataPoint.y - d_currentCentroids[clusterId].y,2);
            distanceToCentroid += pow(dataPoint.z - d_currentCentroids[clusterId].z,2);
            distanceToCentroid = sqrt(distanceToCentroid);
            minDistance = fmin(minDistance, distanceToCentroid);
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

        int *clusterSizes = new int[numClusters];
        for(int i=0; i < numClusters; i++){
            clusterSizes[i]=0;
        }
        int* d_clusterSizes;
        CUDA_CHECK_ERROR(cudaMalloc((void**)&d_clusterSizes, numClusters * sizeof(int)));
        CUDA_CHECK_ERROR(cudaMemcpy(d_clusterSizes, clusterSizes, numClusters * sizeof(int), cudaMemcpyHostToDevice));
        calculateClusterSizesKernel<<<gridSize, blockSize>>>(numPoints, d_assignments, d_clusterSizes);
        calculateNewCentroidsKernel<<<gridSize, blockSize>>>(numPoints, d_points, d_assignments, d_currentCentroids, d_clusterSizes);
        calculateFinalCentroidsKernel<<<gridSize, blockSize>>>(d_currentCentroids, d_clusterSizes, numClusters);
        CUDA_CHECK_ERROR(cudaMemcpy(clusterSizes, d_clusterSizes, numClusters * sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK_ERROR(cudaFree(d_clusterSizes));

        //Update CurrentSSE
        currentSSE = 0.0;
        double* d_currentSSE;
        CUDA_CHECK_ERROR(cudaMalloc((void**)&d_currentSSE, sizeof(double)));
        double hostSSE = 0.0;
        CUDA_CHECK_ERROR(cudaMemcpy(d_currentSSE, &hostSSE, sizeof(double), cudaMemcpyHostToDevice));
        calculateSSEKernel<<<gridSize, blockSize>>>(d_points, numPoints, d_assignments, d_currentCentroids, d_currentSSE);
        CUDA_CHECK_ERROR(cudaMemcpy(&hostSSE, d_currentSSE, sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK_ERROR(cudaFree(d_currentSSE));
        currentSSE = hostSSE/numPoints;
        if(printConsole){
            std::cout <<"Current SSE = " << currentSSE << "" << std::endl;
        }
        //update iteration
        CUDA_CHECK_ERROR(cudaFree(d_assignments));
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

__global__ void calculateSSEKernel(Point* points, int numPoints, int* assignments, Point* newCentroids, double* currentSSE) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < numPoints) {
        int assignment = assignments[idx];
        Point point = points[idx];
        Point assignedCentroid = newCentroids[assignment];
        double distance = 0;
        distance += pow(point.x - assignedCentroid.x, 2);
        distance += pow(point.y - assignedCentroid.y, 2);
        distance += pow(point.z - assignedCentroid.z, 2);

        atomicAdd(currentSSE, distance);
    }
}

__global__ void calculateClusterSizesKernel(int numPoints, const int* assignments, int* clusterSizes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numPoints) {
        atomicAdd(&clusterSizes[assignments[idx]], 1);
    }
}

__global__ void calculateNewCentroidsKernel(int numPoints, Point* points, const int* assignments, Point* newCentroids, int* clusterSizes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numPoints) {
        int assignment = assignments[idx];
        atomicAdd(&newCentroids[assignment].x, points[idx].x);
        atomicAdd(&newCentroids[assignment].y, points[idx].y);
        atomicAdd(&newCentroids[assignment].z, points[idx].z);
    }
}

__global__ void calculateFinalCentroidsKernel(Point* newCentroids, const int* clusterSizes, int numClusters) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numClusters && clusterSizes[idx] > 0) {
        newCentroids[idx].x /= clusterSizes[idx];
        newCentroids[idx].y /= clusterSizes[idx];
        newCentroids[idx].z /= clusterSizes[idx];
    }
}

__global__ void vectorAdd(int* a, int* b, int* c, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        c[tid] = a[tid] + b[tid];
    }
}

__host__ void kickstartGPUCUDA(){
    int n =1000;
    int size = n * sizeof(int);
    int* h_a, * h_b, * h_c;
    int* d_a, * d_b, * d_c;
    h_a = (int*)malloc(size);
    h_b = (int*)malloc(size);
    h_c = (int*)malloc(size);
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);
    for (int i = 0; i < n; ++i) {
        h_a[i] = i;
        h_b[i] = 2 * i;
    }
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);
    int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;
    vectorAdd<<<grid_size, block_size>>>(d_a, d_b, d_c, n);
    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(h_a);
    free(h_b);
    free(h_c);
}

