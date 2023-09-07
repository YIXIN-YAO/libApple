#include "opencv2/opencv.hpp"

#include "common.h"

using namespace std;
static const char *CLASS_NAME = "apple";
static const float BOX_COLOR[3] = {0.000, 0.447, 0.741};

void draw_objects(cv::Mat image, const std::vector<Object> &objects, double prob_threshold) {

    for (const auto &obj : objects)
    {
        if (obj.prob < prob_threshold)
        {
            continue;
        }
        cv::Scalar color = cv::Scalar(BOX_COLOR[0], BOX_COLOR[1], BOX_COLOR[2]);
        float c_mean = (float) cv::mean(color)[0];
        cv::Scalar txt_color;
        if (c_mean > 0.5)
        {
            txt_color = cv::Scalar(0, 0, 0);
        }
        else
        {
            txt_color = cv::Scalar(255, 255, 255);
        }

        cv::rectangle(image, obj.rect, color * 255, 2);

        char text[256];
        sprintf(text, "%s %.1f%% [x:%.3f y:%.3f z:%.3f]",
                CLASS_NAME, obj.prob * 100, obj.world.x, obj.world.y, obj.world.z);

        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.4, 1, &baseLine);

        cv::Scalar txt_bk_color = color * 0.7 * 255;

        int x = (int) obj.rect.x;
        int y = (int) obj.rect.y + 1;
        if (y > image.rows)
            y = image.rows;
        cv::rectangle(image, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
                      txt_bk_color, -1);

        cv::putText(image, text, cv::Point(x, y + label_size.height),
                    cv::FONT_HERSHEY_SIMPLEX, 0.4, txt_color, 1);
    }

}

//获取深度像素对应长度单位（米）的换算比例
float get_depth_scale(const rs2::device &dev) {
    // Go over the device's sensors
    for (rs2::sensor &sensor : dev.query_sensors())
    {
        // Check if the sensor if a depth sensor
        if (rs2::depth_sensor dpt = sensor.as<rs2::depth_sensor>())
        {
            return dpt.get_depth_scale();
        }
    }
    throw std::runtime_error("Device does not have a depth sensor");
}

float depth_filter(const cv::Mat &image_depth, const cv::Mat &mask, int x, int y, int window_size_x, int window_size_y,
                   float scale) {
    int row = image_depth.rows;
    int col = image_depth.cols;

    float vals = 0;
    int nums = 0;
    for (int x_ = x - window_size_x / 2; x_ <= x + window_size_x / 2; x_++)
    {
        for (int y_ = y - window_size_y / 2; y_ <= y + window_size_y / 2; y_++)
        {
            if (x_ < 0 || x_ >= col || y_ < 0 || y_ >= row || mask.at<uint8_t>(y_, x_) > 127)
            {
                continue;
            }

            float val = (float) image_depth.at<uint16_t>(y_, x_) * scale;
            if (val < 0.1 || val > 2)
            {
                continue;
            }

            vals += val;
            nums += 1;
        }
    }
    if (nums <= 0)
    {
        return 0.0;
    }
    return vals / (float) nums;


}




