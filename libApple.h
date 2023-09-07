#ifndef YOLOX_LIBAPPLE_H
#define YOLOX_LIBAPPLE_H

#include <vector>

typedef struct MPoint3f {
    float x = 0.0;
    float y = 0.0;
    float z = 0.0;

    MPoint3f(float x, float y, float z) : x(x), y(y), z(z) {}
} MPoint3f;




typedef struct CUBE {
    MPoint3f v_point;
    MPoint3f l_point;
    MPoint3f r_point;

    CUBE(const MPoint3f &vPoint,
         const MPoint3f &lPoint,
         const MPoint3f &rPoint) : v_point(vPoint),
                                  l_point(lPoint),
                                  r_point(rPoint) {}
} Cube;


typedef struct world_coordinates {
    double x = 0;
    double y = 0;
    double z = 0;
    double prob = 0.0;

    world_coordinates() : x(0), y(0), z(0), prob(0) {}

    world_coordinates(double _x, double _y, double _z, double _p) : x(_x), y(_y), z(_z), prob(_p) {}

} WORLD_COORDINATES;

typedef struct Image {
    unsigned char *data = nullptr;
    int row = 0;
    int col = 0;

    ~Image() {
        delete[] data;
    }
} Image;


bool runtime_start(const char *model,const char* device);
int detect_single_frame(const char* device,std::vector<WORLD_COORDINATES> &data_vec, Image &image);
float detect_tree_distance(const char * device);
int stop_cam(const char* device);
int runtime_stop();



#endif //YOLOX_LIBAPPLE_H
