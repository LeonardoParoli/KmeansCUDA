#include <random>
#include "Kluster.h"

Kluster::Kluster()= default;

Kluster::~Kluster() = default;

Point *Kluster::getCentroid() {
    return &centroid;
}

int Kluster::getSize() const{
    return points.size();
}

std::vector<Point*>* Kluster::getPoints() {
    return &points;
}

void Kluster::addPoint(Point *point) {
     points.push_back(point);
}

void Kluster::resetCluster() {
    points.clear();
    centroid = {-1,-1,-1};
}

void Kluster::clearPoints() {
    points.clear();
}

void Kluster::updateCentroid() {
    if(points.size() > 0){
        Point newCentroid = {0.0f,0.0f,0.0f};
        for(Point* point : points){
            newCentroid.x += point->x;
            newCentroid.y += point->y;
            newCentroid.z += point->z;
        }
        newCentroid.x = newCentroid.x / points.size();
        newCentroid.y = newCentroid.y / points.size();
        newCentroid.z = newCentroid.z / points.size();
        this->centroid = newCentroid;
    }else{
        Point newCentroid = {0.0f,0.0f,0.0f};
        std::random_device rd;
        std::mt19937 rng(rd());
        std::uniform_real_distribution<double> dist(0.0, 1000.0);
        newCentroid.x = dist(rng);
        newCentroid.y = dist(rng);
        newCentroid.z = dist(rng);
        this->centroid = newCentroid;
    }
}

void Kluster::setCentroid(Point *centroid) {
    this->centroid = *centroid;
}
