#ifndef KMEANSSEQUENTIAL_POINT_H
#define KMEANSSEQUENTIAL_POINT_H

#include <cmath>

struct Point{
    Point(double x, double y, double z) {
        this->x=x;
        this->y=y;
        this->z=z;
    }

    Point(){
        this->x=0;
        this->y=0;
        this->z=0;
    }

    double x;
    double y;
    double z;

    double calculateDistance(const Point b){
        double distance = 0.0;
        distance += pow(this->x - b.x, 2);
        distance += pow(this->y - b.y, 2);
        distance += pow(this->z - b.z, 2);
        return sqrt(distance);
    };
};

#endif //KMEANSSEQUENTIAL_POINT_H
