#ifndef KMEANSSEQUENTIAL_KLUSTER_H
#define KMEANSSEQUENTIAL_KLUSTER_H


#include <vector>
#include "Point.h"

class Kluster{
    private:
        std::vector<Point*> points;
        Point centroid = Point(-1,-1,-1);

    public:
        Kluster();
        ~Kluster();
        Point *getCentroid();
        void setCentroid(Point *centroid);
        std::vector<Point*>* getPoints();
        int getSize() const;
        void addPoint(Point *point);
        void resetCluster();
        void clearPoints();
        void updateCentroid();

};

#endif //KMEANSSEQUENTIAL_KLUSTER_H
