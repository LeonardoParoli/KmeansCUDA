#include <iostream>
#include <random>
#include "KmeansParallelCUDASolver.h"
#include "KmeansCUDA.cuh"

Point *KmeansParallelCUDASolver::getPoints() {
    return points;
}
Kluster *KmeansParallelCUDASolver::getClusters() {
    return clusters;
}
KmeansParallelCUDASolver::~KmeansParallelCUDASolver() {
    delete[] clusters;
}
KmeansParallelCUDASolver::KmeansParallelCUDASolver(Point* workPoints, int numPoints, int numClusters, Point *selectedCentroids) {
    this->points= workPoints;
    this->numPoints= numPoints;
    this->numClusters=numClusters;
    this->selectedCentroids = selectedCentroids;
    auto *tempClusters= new Kluster[numClusters];
    for(int i =0; i < numClusters; i++){
        tempClusters[i] = Kluster();
        Point centroid = {selectedCentroids[i].x, selectedCentroids[i].y, selectedCentroids[i].z};
        tempClusters[i].setCentroid(&centroid);
    }
    this->clusters = tempClusters;
}

void KmeansParallelCUDASolver::solve(bool printConsole) {
    //CUDA parameters
    int blockSize = 256;
    int gridSize = (numPoints + blockSize - 1) / blockSize;

    //transforming points from AoS to SoA
    Points soaPoints = transformAoStoSoA(points,numPoints, gridSize, blockSize);

    //Computing Max SSE
    double maxSSE = 0;
    // AoS method -> improved by calculateMAXSSE_SOA
    //calculateMaxSSE(points, selectedCentroids, numPoints,numClusters,gridSize,blockSize,maxSSE);
    calculateMaxSSE_SOA(soaPoints, selectedCentroids, numPoints,numClusters,gridSize,blockSize,maxSSE);
    maxSSE= maxSSE/numPoints;
    if (printConsole) {
        std::cout << "Max SSE = " << maxSSE << "" << std::endl;
    }

    //Starting Kmeans clustering
    //Aos method -> improved by KmeansCycle_SOA
    //clusters = kmeansCycle(points,numPoints,selectedCentroids,numClusters,maxSSE,printConsole);
    clusters = kmeansCycle_SOA(soaPoints,numPoints,selectedCentroids,numClusters,maxSSE,printConsole);
}

Point *KmeansParallelCUDASolver::getSelectedCentroids() {
    return selectedCentroids;
}

void KmeansParallelCUDASolver::kickstartGPU() {
    kickstartGPUCUDA();
}

