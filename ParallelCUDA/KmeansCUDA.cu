#include "KmeansCUDA.cuh"
#include <iostream>
#include <random>
#include <curand_kernel.h>

static void CheckCudaErrorAux(const char*, unsigned, const char*, cudaError_t);
static void CheckCudaErrorAux(const char* file, unsigned line, const char* statement, cudaError_t err) {
    if (err == cudaSuccess)
        return;
    std::cerr << statement << " returned " << cudaGetErrorString(err) << "(" << err << ") at " << file << ":" << line << std::endl;
    exit(1);
}
#define CUDA_CHECK_ERROR(value) CheckCudaErrorAux(__FILE__,__LINE__, #value, value)

__global__ void transformAoaToSoa(const Point* d_aoaPoints, int numPoints, Points d_soaPoints) {
    d_soaPoints.y = d_soaPoints.x +numPoints;
    d_soaPoints.z = d_soaPoints.y + numPoints;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = tid; i < numPoints; i += stride) {
        d_soaPoints.x[i] = d_aoaPoints[i].x;
        d_soaPoints.y[i] = d_aoaPoints[i].y;
        d_soaPoints.z[i] = d_aoaPoints[i].z;
    }
}
__host__ Points transformAoStoSoA(Point* points, int numPoints, int gridSize, int blockSize) {
    double* contiguousMemory = new double[numPoints*3];
    Points h_soaPoints = {contiguousMemory, contiguousMemory+numPoints, contiguousMemory+numPoints*2};
    Point* d_aoaPoints;
    CUDA_CHECK_ERROR(cudaMalloc((void**)&d_aoaPoints, numPoints * sizeof(Point)));
    CUDA_CHECK_ERROR(cudaMemcpy(d_aoaPoints, points, numPoints * sizeof(Point), cudaMemcpyHostToDevice));
    Points d_soaPoints;
    CUDA_CHECK_ERROR(cudaMalloc((void**)&(d_soaPoints.x), numPoints * sizeof(double)*3));

    // Launch kernel to transform AoA to SoA
    transformAoaToSoa<<<gridSize, blockSize>>>(d_aoaPoints, numPoints, d_soaPoints);
    CUDA_CHECK_ERROR(cudaMemcpy(h_soaPoints.x, d_soaPoints.x, numPoints * sizeof(double)*3, cudaMemcpyDeviceToHost));

    CUDA_CHECK_ERROR(cudaFree(d_aoaPoints));
    CUDA_CHECK_ERROR(cudaFree(d_soaPoints.x));
    return h_soaPoints;
}
__global__ void CUDAcalculateMaxSSE_SOA(Points d_soaPoints, Point* d_currentCentroids, double* d_maxSSE, int numPoints, int numClusters){
    d_soaPoints.y = d_soaPoints.x +numPoints;
    d_soaPoints.z = d_soaPoints.y + numPoints;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int pointsPerThread = (numPoints + blockDim.x - 1) / blockDim.x;
    int startPoint = tid * pointsPerThread;
    int endPoint = min(startPoint + pointsPerThread, numPoints);

    double sse = 0.0;
    for (int pointIndex = startPoint; pointIndex < endPoint; pointIndex++) {
        double minDistance = 0;
        minDistance += pow(d_soaPoints.x[pointIndex] - d_currentCentroids[0].x, 2);
        minDistance += pow(d_soaPoints.y[pointIndex] - d_currentCentroids[0].y, 2);
        minDistance += pow(d_soaPoints.z[pointIndex] - d_currentCentroids[0].z, 2);
        minDistance = sqrt(minDistance);

        for (int clusterId = 1; clusterId < numClusters; clusterId++) {
            double distanceToCentroid = 0;
            distanceToCentroid += pow(d_soaPoints.x[pointIndex] - d_currentCentroids[clusterId].x, 2);
            distanceToCentroid += pow(d_soaPoints.y[pointIndex] - d_currentCentroids[clusterId].y, 2);
            distanceToCentroid += pow(d_soaPoints.z[pointIndex] - d_currentCentroids[clusterId].z, 2);
            distanceToCentroid = sqrt(distanceToCentroid);
            minDistance = fmin(minDistance, distanceToCentroid);
        }
        sse += minDistance * minDistance;
    }
    atomicAdd(d_maxSSE, sse);
}
__host__ void calculateMaxSSE_SOA(Points soaPoints,Point* selectedCentroids, int numPoints, int numClusters, int gridSize, int blockSize, double& maxSSE){
    Points d_soaPoints;
    CUDA_CHECK_ERROR(cudaMalloc((void**)&(d_soaPoints.x), numPoints * sizeof(double)*3));
    CUDA_CHECK_ERROR(cudaMemcpy(d_soaPoints.x, soaPoints.x, numPoints * sizeof(double)*3, cudaMemcpyHostToDevice));
    Point* d_currentCentroids;
    CUDA_CHECK_ERROR(cudaMalloc((void**)&d_currentCentroids, numClusters * sizeof(Point)));
    double* d_maxSSE;
    CUDA_CHECK_ERROR(cudaMalloc((void**)&d_maxSSE, sizeof(double)));
    CUDA_CHECK_ERROR(cudaMemcpy(d_currentCentroids, selectedCentroids, numClusters * sizeof(Point), cudaMemcpyHostToDevice));
    CUDAcalculateMaxSSE_SOA<<<gridSize, blockSize>>>(d_soaPoints, d_currentCentroids, d_maxSSE, numPoints, numClusters);
    CUDA_CHECK_ERROR(cudaMemcpy(&maxSSE, d_maxSSE, sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK_ERROR(cudaFree(d_soaPoints.x));
    CUDA_CHECK_ERROR(cudaFree(d_currentCentroids));
    CUDA_CHECK_ERROR(cudaFree(d_maxSSE));
}

__host__ Kluster* kmeansCycle_SOA(Points soaPoints, int numPoints,Point* selectedCentroids, int numClusters, float maxSSE, bool printConsole) {
    //CUDA parameters:
    int blockSize=256;
    int gridSize = (numPoints + blockSize - 1) / blockSize;

    Points d_soaPoints;
    CUDA_CHECK_ERROR(cudaMalloc((void**)&(d_soaPoints.x), numPoints * sizeof(double)*3));
    CUDA_CHECK_ERROR(cudaMemcpy(d_soaPoints.x, soaPoints.x, numPoints * sizeof(double)*3, cudaMemcpyHostToDevice));
    Point* d_currentCentroids;
    CUDA_CHECK_ERROR(cudaMalloc((void **) &d_currentCentroids, numClusters * sizeof(Point)));
    CUDA_CHECK_ERROR(cudaMemcpy(d_currentCentroids, selectedCentroids, numClusters * sizeof(Point), cudaMemcpyHostToDevice));

    float currentSSE=maxSSE;
    float previousSSE = 1e20;
    int iteration = 0;
    int *d_assignments;
    CUDA_CHECK_ERROR(cudaMalloc((void **) &d_assignments, numPoints * sizeof(int)));
    double* d_currentSSE;
    CUDA_CHECK_ERROR(cudaMalloc((void**)&d_currentSSE, sizeof(double)));
    int* d_clusterSizes;
    CUDA_CHECK_ERROR(cudaMalloc((void**)&d_clusterSizes, numClusters * sizeof(int)));

    while ((previousSSE - currentSSE) >= 0.01 && iteration < 10000) {
        previousSSE = currentSSE;

        //assigning points to cluster
        int sharedMemorySize = 3 * blockSize * sizeof(double);
        assignPointsToClusters_SOA<<<gridSize, blockSize, sharedMemorySize>>>(d_soaPoints, numPoints, numClusters, d_currentCentroids, d_assignments);

        //updating centroids
        auto* newCentroids = new Point[numClusters];
        for(int i=0; i < numClusters; i++){
            newCentroids[i] = {0.0,0.0,0.0};
        }
        CUDA_CHECK_ERROR(cudaMemcpy(d_currentCentroids, newCentroids, numClusters * sizeof(Point), cudaMemcpyHostToDevice));
        CUDA_CHECK_ERROR(cudaMemset(d_clusterSizes, 0, numClusters * sizeof(int)));
        calculateNewCentroidsKernel_SOA<<<gridSize, blockSize>>>(numPoints, d_soaPoints, d_assignments, d_currentCentroids, d_clusterSizes);
        calculateFinalCentroidsKernel<<<gridSize, blockSize>>>(d_currentCentroids, d_clusterSizes, numClusters);

        //Update CurrentSSE
        currentSSE = 0.0;
        double hostSSE = 0.0;
        CUDA_CHECK_ERROR(cudaMemcpy(d_currentSSE, &hostSSE, sizeof(double), cudaMemcpyHostToDevice));
        calculateSSEKernel_SOA<<<gridSize, blockSize>>>(d_soaPoints, numPoints, d_assignments, d_currentCentroids, d_currentSSE);
        cudaDeviceSynchronize();
        CUDA_CHECK_ERROR(cudaMemcpy(&hostSSE, d_currentSSE, sizeof(double), cudaMemcpyDeviceToHost));
        currentSSE = hostSSE/numPoints;
        if(printConsole){
            std::cout <<"Current SSE = " << currentSSE << "" << std::endl;
        }
        //update iteration
        iteration++;
    }

    //saving results
    CUDA_CHECK_ERROR(cudaFree(d_soaPoints.x));
    //CUDA_CHECK_ERROR(cudaFree(d_soaPoints.x));
    CUDA_CHECK_ERROR(cudaFree(d_currentSSE));
    CUDA_CHECK_ERROR(cudaFree(d_clusterSizes));
    auto *newCentroids = new Point[numClusters];
    CUDA_CHECK_ERROR(cudaMemcpy(newCentroids, d_currentCentroids, numClusters * sizeof(Point), cudaMemcpyDeviceToHost));
    CUDA_CHECK_ERROR(cudaFree(d_currentCentroids));
    int *assignments = new int[numPoints];
    CUDA_CHECK_ERROR(cudaMemcpy(assignments, d_assignments, numPoints * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK_ERROR(cudaFree(d_assignments));

    //saving iteration on Klusters
    auto * finalClusters = new Kluster[numClusters];
    for(int j = 0; j < numClusters; j++){
        finalClusters[j] = Kluster();
    }
    for(int i = 0; i < numPoints; i++){
        auto* point = new Point(soaPoints.x[i],soaPoints.y[i],soaPoints.z[i]);
        finalClusters[assignments[i]].addPoint(point);
    }
    for(int i = 0; i < numClusters; i++) {
        finalClusters[i].setCentroid(&newCentroids[i]);
    }
    delete[] assignments;
    return finalClusters;
}
__global__ void assignPointsToClusters_SOA(Points d_soaPoints, int numPoints, int numClusters, const Point* d_currentCentroids, int* d_assignments) {
    extern __shared__ double sharedPoints[];
    d_soaPoints.y = d_soaPoints.x +numPoints;
    d_soaPoints.z = d_soaPoints.y + numPoints;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int i = tid; i < numPoints; i += stride) {
        double x = d_soaPoints.x[i];
        double y = d_soaPoints.y[i];
        double z = d_soaPoints.z[i];

        // Load shared memory with current point's components
        sharedPoints[threadIdx.x] = x;
        sharedPoints[threadIdx.x + blockDim.x] = y;
        sharedPoints[threadIdx.x + 2 * blockDim.x] = z;
        __syncthreads(); //SYNCH

        x = sharedPoints[threadIdx.x];
        y = sharedPoints[threadIdx.x + blockDim.x];
        z = sharedPoints[threadIdx.x + 2 * blockDim.x];
        double minDistance = DBL_MAX;
        int nearestCluster = 0;
        for (int j = 0; j < numClusters; ++j) {
            double distance = 0.0;
            distance += pow(x - d_currentCentroids[j].x, 2);
            distance += pow(y - d_currentCentroids[j].y, 2);
            distance += pow(z - d_currentCentroids[j].z, 2);
            distance = sqrt(distance);
            if (distance < minDistance) {
                minDistance = distance;
                nearestCluster = j;
            }
        }
        d_assignments[i] = nearestCluster;
    }
}

__global__ void calculateNewCentroidsKernel_SOA(int numPoints, Points d_soaPoints, const int* d_assignments, Point* d_newCentroids, int* d_clusterSizes){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numPoints) {
        int assignment = d_assignments[idx];
        atomicAdd(&d_newCentroids[assignment].x, d_soaPoints.x[idx]);
        atomicAdd(&d_newCentroids[assignment].y, d_soaPoints.x[idx+numPoints]);
        atomicAdd(&d_newCentroids[assignment].z, d_soaPoints.x[idx+numPoints*2]);
        atomicAdd(&d_clusterSizes[d_assignments[idx]], 1);
    }
}

__global__ void calculateSSEKernel_SOA(Points d_soaPoints, int numPoints, const int* d_assignments, Point* d_newCentroids, double* d_currentSSE){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numPoints) {
        int assignment = d_assignments[idx];
        double x = d_soaPoints.x[idx];
        double y = d_soaPoints.x[idx+ numPoints];
        double z = d_soaPoints.x[idx+ numPoints*2];

        Point assignedCentroid = d_newCentroids[assignment];
        double distance = 0;
        distance += pow(x - assignedCentroid.x, 2);
        distance += pow(y - assignedCentroid.y, 2);
        distance += pow(z - assignedCentroid.z, 2);

        atomicAdd(d_currentSSE, distance);
    }
}

__global__ void calculateFinalCentroidsKernel(Point* newCentroids, const int* clusterSizes, int numClusters) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numClusters && clusterSizes[idx] > 0) {
        newCentroids[idx].x /= clusterSizes[idx];
        newCentroids[idx].y /= clusterSizes[idx];
        newCentroids[idx].z /= clusterSizes[idx];
    }else if(idx < numClusters){
        curandState state;
        curand_init(1001, idx, 0, &state);
        unsigned int randomValue = curand(&state) % 1001;
        newCentroids[idx].x = curand(&state) % 1001;
        newCentroids[idx].y = curand(&state) % 1001;
        newCentroids[idx].z = curand(&state) % 1001;
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


/*
__global__ void transformAoaToSoa(const Point* d_aoaPoints, int numPoints, Points d_soaPoints) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = tid; i < numPoints; i += stride) {
        d_soaPoints.x[i] = d_aoaPoints[i].x;
        d_soaPoints.y[i] = d_aoaPoints[i].y;
        d_soaPoints.z[i] = d_aoaPoints[i].z;
    }
}
__host__ Points transformAoStoSoA(Point* points, int numPoints, int gridSize, int blockSize) {
    Points h_soaPoints = {new double[numPoints], new double[numPoints], new double[numPoints]};
    Point* d_aoaPoints;
    CUDA_CHECK_ERROR(cudaMalloc((void**)&d_aoaPoints, numPoints * sizeof(Point)));
    CUDA_CHECK_ERROR(cudaMemcpy(d_aoaPoints, points, numPoints * sizeof(Point), cudaMemcpyHostToDevice));
    Points d_soaPoints;
    CUDA_CHECK_ERROR(cudaMalloc((void**)&(d_soaPoints.x), numPoints * sizeof(double)));
    CUDA_CHECK_ERROR(cudaMalloc((void**)&(d_soaPoints.y), numPoints * sizeof(double)));
    CUDA_CHECK_ERROR(cudaMalloc((void**)&(d_soaPoints.z), numPoints * sizeof(double)));

    // Launch kernel to transform AoA to SoA
    transformAoaToSoa<<<gridSize, blockSize>>>(d_aoaPoints, numPoints, d_soaPoints);
    CUDA_CHECK_ERROR(cudaMemcpy(h_soaPoints.x, d_soaPoints.x, numPoints * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK_ERROR(cudaMemcpy(h_soaPoints.y, d_soaPoints.y, numPoints * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK_ERROR(cudaMemcpy(h_soaPoints.z, d_soaPoints.z, numPoints * sizeof(double), cudaMemcpyDeviceToHost));

    CUDA_CHECK_ERROR(cudaFree(d_aoaPoints));
    CUDA_CHECK_ERROR(cudaFree(d_soaPoints.x));
    CUDA_CHECK_ERROR(cudaFree(d_soaPoints.y));
    CUDA_CHECK_ERROR(cudaFree(d_soaPoints.z));
    return h_soaPoints;
}
__global__ void CUDAcalculateMaxSSE_SOA(Points d_soaPoints, Point* d_currentCentroids, double* d_maxSSE, int numPoints, int numClusters){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int pointsPerThread = (numPoints + blockDim.x - 1) / blockDim.x;
    int startPoint = tid * pointsPerThread;
    int endPoint = min(startPoint + pointsPerThread, numPoints);

    double sse = 0.0;
    for (int pointIndex = startPoint; pointIndex < endPoint; pointIndex++) {
        double minDistance = 0;
        minDistance += pow(d_soaPoints.x[pointIndex] - d_currentCentroids[0].x, 2);
        minDistance += pow(d_soaPoints.y[pointIndex] - d_currentCentroids[0].y, 2);
        minDistance += pow(d_soaPoints.z[pointIndex] - d_currentCentroids[0].z, 2);
        minDistance = sqrt(minDistance);

        for (int clusterId = 1; clusterId < numClusters; clusterId++) {
            double distanceToCentroid = 0;
            distanceToCentroid += pow(d_soaPoints.x[pointIndex] - d_currentCentroids[clusterId].x, 2);
            distanceToCentroid += pow(d_soaPoints.y[pointIndex] - d_currentCentroids[clusterId].y, 2);
            distanceToCentroid += pow(d_soaPoints.z[pointIndex] - d_currentCentroids[clusterId].z, 2);
            distanceToCentroid = sqrt(distanceToCentroid);
            minDistance = fmin(minDistance, distanceToCentroid);
        }
        sse += minDistance * minDistance;
    }
    atomicAdd(d_maxSSE, sse);
}
__host__ void calculateMaxSSE_SOA( Points soaPoints,Point* selectedCentroids, int numPoints, int numClusters, int gridSize, int blockSize, double& maxSSE){
    Points d_soaPoints;
    CUDA_CHECK_ERROR(cudaMalloc((void**)&(d_soaPoints.x), numPoints * sizeof(double)));
    CUDA_CHECK_ERROR(cudaMalloc((void**)&(d_soaPoints.y), numPoints * sizeof(double)));
    CUDA_CHECK_ERROR(cudaMalloc((void**)&(d_soaPoints.z), numPoints * sizeof(double)));
    CUDA_CHECK_ERROR(cudaMemcpy(d_soaPoints.x, soaPoints.x, numPoints * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK_ERROR(cudaMemcpy(d_soaPoints.y, soaPoints.y, numPoints * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK_ERROR(cudaMemcpy(d_soaPoints.z, soaPoints.z, numPoints * sizeof(double), cudaMemcpyHostToDevice));
    Point* d_currentCentroids;
    CUDA_CHECK_ERROR(cudaMalloc((void**)&d_currentCentroids, numClusters * sizeof(Point)));
    double* d_maxSSE;
    CUDA_CHECK_ERROR(cudaMalloc((void**)&d_maxSSE, sizeof(double)));
    CUDA_CHECK_ERROR(cudaMemcpy(d_currentCentroids, selectedCentroids, numClusters * sizeof(Point), cudaMemcpyHostToDevice));
    CUDAcalculateMaxSSE_SOA<<<gridSize, blockSize>>>(d_soaPoints, d_currentCentroids, d_maxSSE, numPoints, numClusters);
    CUDA_CHECK_ERROR(cudaMemcpy(&maxSSE, d_maxSSE, sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK_ERROR(cudaFree(d_soaPoints.x));
    CUDA_CHECK_ERROR(cudaFree(d_soaPoints.y));
    CUDA_CHECK_ERROR(cudaFree(d_soaPoints.z));
    CUDA_CHECK_ERROR(cudaFree(d_currentCentroids));
    CUDA_CHECK_ERROR(cudaFree(d_maxSSE));
}

__host__ Kluster* kmeansCycle_SOA(Points soaPoints, int numPoints,Point* selectedCentroids, int numClusters, float maxSSE, bool printConsole) {
    //CUDA parameters:
    int blockSize=256;
    int gridSize = (numPoints + blockSize - 1) / blockSize;

    Points d_soaPoints;
    CUDA_CHECK_ERROR(cudaMalloc((void**)&(d_soaPoints.x), numPoints * sizeof(double)));
    CUDA_CHECK_ERROR(cudaMalloc((void**)&(d_soaPoints.y), numPoints * sizeof(double)));
    CUDA_CHECK_ERROR(cudaMalloc((void**)&(d_soaPoints.z), numPoints * sizeof(double)));
    CUDA_CHECK_ERROR(cudaMemcpy(d_soaPoints.x, soaPoints.x, numPoints * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK_ERROR(cudaMemcpy(d_soaPoints.y, soaPoints.y, numPoints * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK_ERROR(cudaMemcpy(d_soaPoints.z, soaPoints.z, numPoints * sizeof(double), cudaMemcpyHostToDevice));

    Point* d_currentCentroids;
    CUDA_CHECK_ERROR(cudaMalloc((void **) &d_currentCentroids, numClusters * sizeof(Point)));
    CUDA_CHECK_ERROR(cudaMemcpy(d_currentCentroids, selectedCentroids, numClusters * sizeof(Point), cudaMemcpyHostToDevice));

    float currentSSE=maxSSE;
    float previousSSE = 1e20;
    int iteration = 0;
    int *d_assignments;
    CUDA_CHECK_ERROR(cudaMalloc((void **) &d_assignments, numPoints * sizeof(int)));
    double* d_currentSSE;
    CUDA_CHECK_ERROR(cudaMalloc((void**)&d_currentSSE, sizeof(double)));
    int* d_clusterSizes;
    CUDA_CHECK_ERROR(cudaMalloc((void**)&d_clusterSizes, numClusters * sizeof(int)));
    while ((previousSSE - currentSSE) >= 0.01 && iteration < 10000) {
        previousSSE = currentSSE;

        //assigning points to cluster
        int sharedMemorySize = 3 * blockSize * sizeof(double);
        assignPointsToClusters_SOA<<<gridSize, blockSize, sharedMemorySize>>>(d_soaPoints.x,d_soaPoints.y,d_soaPoints.z, numPoints, numClusters, d_currentCentroids, d_assignments);

        //updating centroids
        auto* newCentroids = new Point[numClusters];
        for(int i=0; i < numClusters; i++){
            newCentroids[i] = {0.0,0.0,0.0};
        }
        CUDA_CHECK_ERROR(cudaMemcpy(d_currentCentroids, newCentroids, numClusters * sizeof(Point), cudaMemcpyHostToDevice));
        CUDA_CHECK_ERROR(cudaMemset(d_clusterSizes, 0, numClusters * sizeof(int)));
        calculateNewCentroidsKernel_SOA<<<gridSize, blockSize>>>(numPoints, d_soaPoints, d_assignments, d_currentCentroids, d_clusterSizes);
        calculateFinalCentroidsKernel<<<gridSize, blockSize>>>(d_currentCentroids, d_clusterSizes, numClusters);

        //Update CurrentSSE
        currentSSE = 0.0;
        double hostSSE = 0.0;
        CUDA_CHECK_ERROR(cudaMemcpy(d_currentSSE, &hostSSE, sizeof(double), cudaMemcpyHostToDevice));
        calculateSSEKernel_SOA<<<gridSize, blockSize>>>(d_soaPoints, numPoints, d_assignments, d_currentCentroids, d_currentSSE);
        cudaDeviceSynchronize();
        CUDA_CHECK_ERROR(cudaMemcpy(&hostSSE, d_currentSSE, sizeof(double), cudaMemcpyDeviceToHost));
        currentSSE = hostSSE/numPoints;
        if(printConsole){
            std::cout <<"Current SSE = " << currentSSE << "" << std::endl;
        }
        //update iteration
        iteration++;
    }

    //saving results
    CUDA_CHECK_ERROR(cudaFree(d_soaPoints.x));
    CUDA_CHECK_ERROR(cudaFree(d_soaPoints.y));
    CUDA_CHECK_ERROR(cudaFree(d_soaPoints.z));
    //CUDA_CHECK_ERROR(cudaFree(d_soaPoints.x));
    CUDA_CHECK_ERROR(cudaFree(d_currentSSE));
    CUDA_CHECK_ERROR(cudaFree(d_clusterSizes));
    auto *newCentroids = new Point[numClusters];
    CUDA_CHECK_ERROR(cudaMemcpy(newCentroids, d_currentCentroids, numClusters * sizeof(Point), cudaMemcpyDeviceToHost));
    CUDA_CHECK_ERROR(cudaFree(d_currentCentroids));
    int *assignments = new int[numPoints];
    CUDA_CHECK_ERROR(cudaMemcpy(assignments, d_assignments, numPoints * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK_ERROR(cudaFree(d_assignments));

    //saving iteration on Klusters
    auto * finalClusters = new Kluster[numClusters];
    for(int j = 0; j < numClusters; j++){
        finalClusters[j] = Kluster();
    }
    for(int i = 0; i < numPoints; i++){
        auto* point = new Point(soaPoints.x[i],soaPoints.y[i],soaPoints.z[i]);
        finalClusters[assignments[i]].addPoint(point);
    }
    for(int i = 0; i < numClusters; i++) {
        finalClusters[i].setCentroid(&newCentroids[i]);
    }
    delete[] assignments;
    return finalClusters;
}
__global__ void assignPointsToClusters_SOA(const double* d_x, const double* d_y, const double* d_z, int numPoints, int numClusters, const Point* d_currentCentroids, int* d_assignments) {
    extern __shared__ double sharedPoints[];
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int i = tid; i < numPoints; i += stride) {
        double x = d_x[i];
        double y = d_y[i];
        double z = d_z[i];

        // Load shared memory with current point's components
        sharedPoints[threadIdx.x] = x;
        sharedPoints[threadIdx.x + blockDim.x] = y;
        sharedPoints[threadIdx.x + 2 * blockDim.x] = z;
        __syncthreads(); //SYNCH

        x = sharedPoints[threadIdx.x];
        y = sharedPoints[threadIdx.x + blockDim.x];
        z = sharedPoints[threadIdx.x + 2 * blockDim.x];
        double minDistance = DBL_MAX;
        int nearestCluster = 0;
        for (int j = 0; j < numClusters; ++j) {
            double distance = 0.0;
            distance += pow(x - d_currentCentroids[j].x, 2);
            distance += pow(y - d_currentCentroids[j].y, 2);
            distance += pow(z - d_currentCentroids[j].z, 2);
            distance = sqrt(distance);
            if (distance < minDistance) {
                minDistance = distance;
                nearestCluster = j;
            }
        }
        d_assignments[i] = nearestCluster;
    }
}

__global__ void calculateNewCentroidsKernel_SOA(int numPoints, Points d_soaPoints, const int* d_assignments, Point* d_newCentroids, int* d_clusterSizes){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numPoints) {
        int assignment = d_assignments[idx];
        atomicAdd(&d_newCentroids[assignment].x, d_soaPoints.x[idx]);
        atomicAdd(&d_newCentroids[assignment].y, d_soaPoints.y[idx]);
        atomicAdd(&d_newCentroids[assignment].z, d_soaPoints.z[idx]);
        atomicAdd(&d_clusterSizes[d_assignments[idx]], 1);
    }
}

__global__ void calculateSSEKernel_SOA(Points d_soaPoints, int numPoints, const int* d_assignments, Point* d_newCentroids, double* d_currentSSE){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numPoints) {
        int assignment = d_assignments[idx];
        double x = d_soaPoints.x[idx];
        double y = d_soaPoints.y[idx];
        double z = d_soaPoints.z[idx];

        Point assignedCentroid = d_newCentroids[assignment];
        double distance = 0;
        distance += pow(x - assignedCentroid.x, 2);
        distance += pow(y - assignedCentroid.y, 2);
        distance += pow(z - assignedCentroid.z, 2);

        atomicAdd(d_currentSSE, distance);
    }
}

__global__ void calculateFinalCentroidsKernel(Point* newCentroids, const int* clusterSizes, int numClusters) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numClusters && clusterSizes[idx] > 0) {
        newCentroids[idx].x /= clusterSizes[idx];
        newCentroids[idx].y /= clusterSizes[idx];
        newCentroids[idx].z /= clusterSizes[idx];
    }else if(idx < numClusters){
        curandState state;
        curand_init(1001, idx, 0, &state);
        unsigned int randomValue = curand(&state) % 1001;
        newCentroids[idx].x = curand(&state) % 1001;
        newCentroids[idx].y = curand(&state) % 1001;
        newCentroids[idx].z = curand(&state) % 1001;
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
*/
//////////////////////////// ^^^^^^^^^^^
//////////////////////////// Array of Structures methods -> improved by SOA methods
//////////////////////////// ^^^^^^^^^^^

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

__host__ Kluster* kmeansCycle(Point* points,int numPoints,Point* selectedCentroids, int numClusters, float maxSSE, bool printConsole) {
    //CUDA parameters:
    int blockSize=256;
    int gridSize = (numPoints + blockSize - 1) / blockSize;

    Point* d_points;
    Point* d_currentCentroids;
    CUDA_CHECK_ERROR(cudaMalloc((void **) &d_points, numPoints * sizeof(Point)));
    CUDA_CHECK_ERROR(cudaMalloc((void **) &d_currentCentroids, numClusters * sizeof(Point)));
    CUDA_CHECK_ERROR(cudaMemcpy(d_points, points, numPoints * sizeof(Point), cudaMemcpyHostToDevice));
    CUDA_CHECK_ERROR(cudaMemcpy(d_currentCentroids, selectedCentroids, numClusters * sizeof(Point), cudaMemcpyHostToDevice));

    float currentSSE=maxSSE;
    float previousSSE = 1e20;
    int iteration = 0;
    int *d_assignments;
    CUDA_CHECK_ERROR(cudaMalloc((void **) &d_assignments, numPoints * sizeof(int)));
    double* d_currentSSE;
    CUDA_CHECK_ERROR(cudaMalloc((void**)&d_currentSSE, sizeof(double)));
    int* d_clusterSizes;
    CUDA_CHECK_ERROR(cudaMalloc((void**)&d_clusterSizes, numClusters * sizeof(int)));
    while ((previousSSE - currentSSE) >= 0.01 && iteration < 10000) {
        previousSSE = currentSSE;

        //assigning points to cluster
        size_t sharedMemorySize = blockSize * sizeof(Point);
        assignPointsToClusters<<<gridSize, blockSize, sharedMemorySize>>>(d_points, numPoints, numClusters, d_currentCentroids, d_assignments);

        //updating centroids
        auto* newCentroids = new Point[numClusters];
        for(int i=0; i < numClusters; i++){
            newCentroids[i] = {0.0,0.0,0.0};
        }
        CUDA_CHECK_ERROR(cudaMemcpy(d_currentCentroids, newCentroids, numClusters * sizeof(Point), cudaMemcpyHostToDevice));
        CUDA_CHECK_ERROR(cudaMemset(d_clusterSizes, 0, numClusters * sizeof(int)));
        calculateNewCentroidsKernel<<<gridSize, blockSize>>>(numPoints, d_points, d_assignments, d_currentCentroids, d_clusterSizes);
        calculateFinalCentroidsKernel<<<gridSize, blockSize>>>(d_currentCentroids, d_clusterSizes, numClusters);

        //Update CurrentSSE
        currentSSE = 0.0;
        double hostSSE = 0.0;
        CUDA_CHECK_ERROR(cudaMemcpy(d_currentSSE, &hostSSE, sizeof(double), cudaMemcpyHostToDevice));
        calculateSSEKernel<<<gridSize, blockSize>>>(d_points, numPoints, d_assignments, d_currentCentroids, d_currentSSE);
        CUDA_CHECK_ERROR(cudaMemcpy(&hostSSE, d_currentSSE, sizeof(double), cudaMemcpyDeviceToHost));
        currentSSE = hostSSE/numPoints;
        if(printConsole){
            std::cout <<"Current SSE = " << currentSSE << "" << std::endl;
        }
        //update iteration
        iteration++;
    }

    //saving results
    CUDA_CHECK_ERROR(cudaFree(d_points));
    CUDA_CHECK_ERROR(cudaFree(d_currentSSE));
    CUDA_CHECK_ERROR(cudaFree(d_clusterSizes));
    auto *newCentroids = new Point[numClusters];
    CUDA_CHECK_ERROR(cudaMemcpy(newCentroids, d_currentCentroids, numClusters * sizeof(Point), cudaMemcpyDeviceToHost));
    CUDA_CHECK_ERROR(cudaFree(d_currentCentroids));
    int *assignments = new int[numPoints];
    CUDA_CHECK_ERROR(cudaMemcpy(assignments, d_assignments, numPoints * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK_ERROR(cudaFree(d_assignments));

    //saving iteration on Klusters
    auto * finalClusters = new Kluster[numClusters];
    for(int j = 0; j < numClusters; j++){
        finalClusters[j] = Kluster();
    }
    for(int i = 0; i < numPoints; i++){
        finalClusters[assignments[i]].addPoint(&points[i]);
    }
    for(int i = 0; i < numClusters; i++) {
        finalClusters[i].setCentroid(&newCentroids[i]);
    }

    delete[] assignments;
    return finalClusters;
}

__global__ void assignPointsToClusters(Point* d_points, int numPoints, int numClusters, Point* d_currentCentroids, int* d_assignments) {
    extern __shared__ Point sharedP[];

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    // Load data into shared memory
    for (int i = tid; i < numPoints; i += stride) {
        sharedP[threadIdx.x] = d_points[i];
        __syncthreads();

        // Perform calculations using sharedPoints
        Point point = sharedP[threadIdx.x];
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
        __syncthreads();
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

__global__ void calculateNewCentroidsKernel(int numPoints, Point* points, const int* assignments, Point* newCentroids, int* clusterSizes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numPoints) {
        int assignment = assignments[idx];
        atomicAdd(&newCentroids[assignment].x, points[idx].x);
        atomicAdd(&newCentroids[assignment].y, points[idx].y);
        atomicAdd(&newCentroids[assignment].z, points[idx].z);
        atomicAdd(&clusterSizes[assignments[idx]], 1);
    }
}
