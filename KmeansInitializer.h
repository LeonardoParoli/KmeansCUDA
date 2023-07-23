#ifndef KMEANSSEQUENTIAL_KMEANSINITIALIZER_H
#define KMEANSSEQUENTIAL_KMEANSINITIALIZER_H

#include "Point.h"

class KmeansInitializer {
    private:
        Point* points;
        Point* realCentroids;
        int numPoints;
        int numClusters;

public:
    KmeansInitializer(int numPoints, int numClusters, double coordinateRange, double clusterRadius);
    ~KmeansInitializer();

    Point *getPoints() const;
    Point *getRealCentroids() const;
    int getNumPoints() const;
    int getNumClusters() const;
};


#endif //KMEANSSEQUENTIAL_KMEANSINITIALIZER_H
