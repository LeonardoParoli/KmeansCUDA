#include <iostream>
#include <random>
#include "KmeansParallelCUDASolver.h"
#include <cfloat>
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

    //Computing Max SSE
    float maxSSE = 0;
    Point* d_points;
    Point* d_currentCentroids;
    float* d_maxSSE;

    calculateMaxSSE(d_points,points, d_currentCentroids,selectedCentroids, d_maxSSE, numPoints, numClusters,gridSize,blockSize,maxSSE);

    maxSSE= maxSSE/numPoints;
    if (printConsole) {
        std::cout << "Max SSE = " << maxSSE << "" << std::endl;
    }

    //Starting Kmeans clustering
    // TODO bring this down into the kmeansIteration to avoid having to cudamalloc everytime
    double currentSSE = maxSSE;
    double previousSSE = 1e20;
    int iteration = 0;
    auto *newCentroids = new Point[numClusters];
    for(int i = 0; i < numClusters; i++){
        newCentroids[i] = selectedCentroids[i];
    }

    while ((previousSSE - currentSSE) >= 0.01 && iteration < 10000) {
        Point* d_iterationPoints;
        Point* d_iterationCentroids;
        previousSSE = currentSSE;
        kmeansIteration(d_iterationPoints,points,numPoints,d_iterationCentroids,newCentroids,numClusters);
        iteration++;
    }

    std::cout << "FINISHED FINISHED FINISHED" << std::endl;
}

Point *KmeansParallelCUDASolver::getSelectedCentroids() {
    return selectedCentroids;
}
