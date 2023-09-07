#ifndef YOLOX_COMMON_H
#define YOLOX_COMMON_H

#include <opencv2/opencv.hpp>
#include <librealsense2/rs.hpp> // Include RealSense Cross Platform API
#include <librealsense2/rsutil.h>

struct Object {
    cv::Rect2f rect;
    int label = 0;
    float prob = 0.0;
    cv::Point3d world;
};

void draw_objects(cv::Mat image, const std::vector<Object> &objects, double prob_threshold = 0.3);

float get_depth_scale(const rs2::device &dev);

float depth_filter(const cv::Mat &image_depth, const cv::Mat &mask, int x, int y,
                   int window_size_x, int window_size_y, float scale = 0.001);

#endif //YOLOX_COMMON_H
